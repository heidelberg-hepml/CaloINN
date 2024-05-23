import math
from operator import le
import re
import numpy as np
import data_util

import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm

from myBlocks import *
from vblinear import VBLinear
from splines.rational_quadratic import RationalQuadraticSpline


import copy
import torch.nn.functional as F

from copy import deepcopy

class Subnet(nn.Module):
    """ This class constructs a subnet for the coupling blocks """

    def __init__(self, layer_classes, size_in, size_out, internal_sizes=None, dropout=0.0, layer_args=None,
                 layer_act="nn.ReLU", layer_norm=None):
        """
            Initializes subnet class.

            Parameters:
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet. If None, set to 2*size
            dropout: dropout chance of the subnet
        """
        super().__init__()
        num_layers = len(layer_classes)
        
        if internal_sizes is None:
            internal_sizes = size_out * 2
            
        if layer_args is None:
            layer_args = [{}] * num_layers
            
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        
        self.layer_list = []
        
        if isinstance(internal_sizes, list):
            assert len(internal_sizes) == num_layers-1
            [size_in] + internal_sizes + [size_out]
        else:
            internal_sizes = [internal_sizes] * (num_layers-1)
            internal_sizes = [size_in] + internal_sizes + [size_out]
            
        
        for n in range(num_layers):
             
            input_dim, output_dim = internal_sizes[n], internal_sizes[n+1]

            self.layer_list.append(layer_classes[n](input_dim, output_dim, **layer_args[n]))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                if layer_norm is not None:
                    self.layer_list.append(eval(layer_norm)(output_dim))
                self.layer_list.append(eval(layer_act)())

        self.layers = nn.Sequential(*self.layer_list)

        final_layer_name = str(len(self.layers) - 1)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data.zero_()

    def forward(self, x):
        return self.layers(x)


class FixedAffineTransform(fm.InvertibleModule):
    '''Fixed transformation according to y = M*x + b.'''

    def __init__(self, dims_in, dims_c=None, M=None, b=None):
        super().__init__(dims_in, dims_c)

        self.M = nn.Parameter(M, requires_grad=False)
        self.M_inv = nn.Parameter(1/M, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False)

        self.jac_det = nn.Parameter(self.M.log().sum(), requires_grad=False)

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0] * self.M + self.b], self.jac_det
        else:
            return [(x[0]-self.b) * self.M_inv], -self.jac_det

    def output_dims(self, input_dims):
        return input_dims


class LogTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if rev:
            z = torch.exp(x) - self.alpha
            jac = torch.sum( x, dim=1)
        else:
            z = torch.log(x + self.alpha)
            jac = - torch.sum( z, dim=1)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def output_dims(self, input_dims):
        return input_dims


class LogitTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, alpha = 0.):
        super().__init__(dims_in, dims_c)
        self.alpha = alpha

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        if not rev:
            x = x*(1-2*self.alpha) + self.alpha
            z = torch.logit(x)
        else:
            z = torch.sigmoid(x)
            z = (z - self.alpha)/(1-2*self.alpha)
        return (z, ), torch.tensor([0.], device=x.device) # jac

    def norm_logit(self, t: torch.Tensor):
         f = lambda x: torch.sum(1/(1+torch.exp(-t-x)), dim=1) - 1 - self.alpha*(t.shape[1] - 2)
         f_ = lambda x: torch.sum(torch.exp(-t-x)/(1+torch.exp(-t-x))**2,dim=1)
         c = torch.zeros((t.shape[0], 1), device=t.device)
         for i in range(8):
             c = c - (f(c)/f_(c))[...,None]
         return t+c

    def output_dims(self, input_dims):
        return input_dims


class CINN(nn.Module):
    """ cINN model """

    def __init__(self, params, data, cond):
        """ Initializes model class.

        Parameters:
        params: Dict containing the network and training parameter
        data: Training data to initialize the norm layer
        cond: Conditions to the training data
        """
        super(CINN, self).__init__()
        self.params = params
        self.num_dim = data.shape[1]

        self.norm_m = None
        self.bayesian = params.get("bayesian", False)
        self.alpha = params.get("alpha", 1e-8)
        self.log_cond = params.get("log_cond", False)

        if self.bayesian:
            self.bayesian_layers = []

        self.initialize_normalization(data, cond)
        self.define_model_architecture(self.num_dim)

    def forward(self, x, c, rev=False, jac=True):
        if self.log_cond:
            c_norm = torch.log(c)
        else:
            c_norm = c
        return self.model.forward(x, c_norm, rev=rev, jac=jac)

    def get_constructor_func(self, params):
        
        """ Returns a function that constructs a subnetwork with the given parameters """
        
        def get_layer_classes(lay_params):
            lays = []
            for n in range(len(lay_params)):
                if lay_params[n] == 'vblinear':
                    lays.append(VBLinear)
                elif lay_params[n] == 'linear':
                    lays.append(nn.Linear)
                else:
                    raise ValueError(f"Unknown layer type {lay_params[n]}")
            return lays
        
        def get_layer_args(params):
            layer_classes = params["layer_classes"]
            layer_args = []
            for n in range(len(layer_classes)):
                n_args = {}
                if layer_classes[n] == "vblinear":
                    if "prior_prec" in params:
                        layer_args["prior_prec"] = params["prior_prec"]
                    if "std_init" in params:
                        layer_args["std_init"] = params["std_init"]
                    if "sigma_fixed" in params:
                        layer_args["sigma_fixed"] = params["sigma_fixed"]
                layer_args.append(n_args)
            return layer_args
        
    
        layer_classes = get_layer_classes(params["layer_classes"])
        layer_args = get_layer_args(params)
        
        def func(x_in, x_out):
            subnet = Subnet(
                    layer_classes,
                    x_in, x_out,
                    internal_sizes = params.get("internal_size"),
                    dropout = params.get("dropout", 0.),
                    layer_args = layer_args,                    
                    layer_norm = params.get("layer_norm", None),
                    layer_act = params.get("layer_act", "nn.ReLU"),)
            if self.bayesian:
                self.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear))
            return subnet
        return func

    def get_coupling_block(self, params):
        """ Returns the class and keyword arguments for different coupling block types """
        constructor_fct = self.get_constructor_func(params)
        permute_soft = params.get("permute_soft")
        coupling_type = params.get("coupling_type", "affine")

        if coupling_type == "affine":
            CouplingBlock = fm.AllInOneBlock
            block_kwargs = {
                            "affine_clamping": params.get("clamping", 5.),
                            "subnet_constructor": constructor_fct,
                            "global_affine_init": 0.92,
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "cubic":
            CouplingBlock = CubicSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init": params.get("bounds_init", 10),
                            "bounds_type": params.get("bounds_type", "SOFTPLUS"),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "rational_quadratic":
            CouplingBlock = RationalQuadraticSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type == "rational_quadratic_freia":
            CouplingBlock = RationalQuadraticSpline
            block_kwargs = {
                            "bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                           }
        elif coupling_type == "MADE":
            CouplingBlock = MADE
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft,
                            "hidden_features": params.get("internal_size"),
                            "num_blocks": params.get("layers_per_block", 3)-2,
                            "dropout": params.get("dropout", 0.)
                           }
        else:
            raise ValueError(f"Unknown Coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def initialize_normalization(self, data, cond):
        """ Calculates the normalization transformation from the training data and stores it. """
        data = torch.clone(data)

        if self.params.get("logit_transformation", True):
            data = data*(1-2*self.alpha) + self.alpha
            data = torch.logit(data)
        
        elif self.params.get("log_transformation", False):
            data = torch.log(data + self.alpha)
        
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = 1 / std
        self.norm_b = - mean/std

        data -= mean
        data /= std

        print('num samples out of bounds:', torch.count_nonzero(torch.max(torch.abs(data), dim=1)[0] > self.params.get("bounds_init", 10)).item())

    def define_model_architecture(self, in_dim):
        """ Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor """

        self.in_dim = in_dim
        if self.norm_m is None:
            self.norm_m = torch.eye(in_dim)
            self.norm_b = torch.zeros(in_dim)

        nodes = [ff.InputNode(in_dim, name="inp")]
        cond_node = ff.ConditionNode(1, name="cond")

        # Add some preprocessing nodes
        if self.params.get("logit_transformation", True):
            nodes.append(ff.Node(
                [nodes[-1].out0],
                LogitTransformation,
                { "alpha": self.alpha },
                name = "inp_log"
            ))
        elif self.params.get("log_transformation", False):
            print("log transformation is used")
            nodes.append(ff.Node(
                [nodes[-1].out0],
                LogTransformation,
                { "alpha": self.alpha },
                name = "inp_log"
            ))
        nodes.append(ff.Node(
            [nodes[-1].out0],
            FixedAffineTransform,
            { "M": self.norm_m, "b": self.norm_b },
            name = "inp_norm"
        ))

        # Add the coupling blocks determined by the params file
        CouplingBlock, block_kwargs = self.get_coupling_block(self.params)
        
        if self.params.get("coupling_type", "affine") != "rational_quadratic_freia":
            
            for i in range(self.params.get("n_blocks", 10)):
                nodes.append(
                    ff.Node(
                        [nodes[-1].out0],
                        CouplingBlock,
                        block_kwargs,
                        conditions = cond_node,
                        name = f"block_{i}"
                    )
                )
                
        # Freia uses a double pass. Each block runs the transformation on both splits. So we need only half of the blocks.
        # Furthermore, we have to add a random permutation block manually.
        else:
            for i in range(self.params.get("n_blocks", 10) // 2):
                nodes.append(
                    ff.Node(
                        [nodes[-1].out0],
                        CouplingBlock,
                        block_kwargs,
                        conditions = cond_node,
                        name = f"block_{i}"
                    )
                )
                nodes.append(
                    ff.Node(
                        [nodes[-1].out0],
                        fm.PermuteRandom,
                        {"seed": None},
                        name = f"permutation_block_{i}"
                    )
                )

        nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)

        self.model = ff.GraphINN(nodes)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"number of parameters: {n_trainable}", flush=True)

    def set_bayesian_std_grad(self, requires_grad):
        for layer in self.bayesian_layers:
            layer.logsig2_w.requires_grad = requires_grad

    def sample_random_state(self):
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, state):
        [layer.import_random_state(s) for layer, s in zip(self.bayesian_layers, state)]

    def get_kl(self):
        return sum(layer.KL() for layer in self.bayesian_layers)

    def enable_map(self):
        for layer in self.bayesian_layers:
            layer.enable_map()

    def disable_map(self):
        for layer in self.bayesian_layers:
            layer.disable_map()
            
    def fix_sigma(self):
        for layer in self.bayesian_layers:
            layer.fix_sigma()
            
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        
    def unfix_sigma(self):
        for layer in self.bayesian_layers:
            layer.unfix_sigma()
            
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        
    def reset_sigma(self):
        for layer in self.bayesian_layers:
            layer.unfix_sigma()
            layer.reset_sigma(self.params.get("std_init", -9))

    def reset_random(self):
        """ samples a new random state for the Bayesian layers """
        for layer in self.bayesian_layers:
            layer.reset_random()

    def sample(self, num_pts, condition, z=None):
        """
            sample from the learned distribution

            Parameters:
            num_pts (int): Number of samples to generate for each given condition
            condition (tensor): Conditions

            Returns:
            tensor[len(condition), num_pts, dims]: Samples 
        """
        if z is None:
            z = torch.normal(0, 1,
                size=(num_pts*condition.shape[0], self.in_dim),
                device=next(self.parameters()).device)
        c = condition.repeat(num_pts,1)
        x, _ = self.forward(z, c, rev=True)
        return x.reshape(num_pts, condition.shape[0], self.in_dim).permute(1,0,2)

    def log_prob(self, x, c):
        """
            evaluate conditional log-likelihoods for given samples and conditions

            Parameters:
            x (tensor): Samples
            c (tensor): Conditions

            Returns:
            tensor: Log-likelihoods
        """
        z, log_jac_det = self.forward(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        return log_prob

    
class LogitTransformationVAE:
    def __init__(self, rev=False, alpha=1.e-6):
        self.alpha = alpha
        self.rev = rev
        
    def forward(self, x):
        if not self.rev:
            x = x*(1-2*self.alpha) + self.alpha
            z = torch.logit(x)
        else:
            z = torch.sigmoid(x)
            z = (z - self.alpha)/(1-2*self.alpha)

        return z
        
    def __call__(self, x):
        return self.forward(x)


class LearnableNorm(nn.Module):
    def __init__(self, num_features, trainable=True, initial_scale=None, initial_bias=None):
        super(LearnableNorm, self).__init__()
        
        num_features = num_features[0][0]
        
        if initial_scale is None:
            initial_scale = torch.ones(num_features)
            
        if initial_bias is None:
            initial_bias = torch.zeros(num_features)
        
        self.scale = nn.Parameter(initial_scale, requires_grad=trainable)
        self.bias = nn.Parameter(initial_bias, requires_grad=trainable)

    def forward(self, x, rev=False):
        if rev:
            return [((x[0] - self.bias[:x[0].shape[1]]) / self.scale[:x[0].shape[1]],)]
        else:
            return [(x[0] * self.scale[:x[0].shape[1]] + self.bias[:x[0].shape[1]],)]
 

class CVAE(nn.Module):
    def __init__(self, input, cond, latent_dim, hidden_sizes, layer_boundaries_detector,
                 particle_type="photon",dataset=1, alpha=1.e-6, beta=1.e-5, gamma=1.e3, 
                 eps=1.e-10, smearing_self=1.0, smearing_share=0.0,
                 threshold=None, sparsity_loss=None, learnable_norm=False):
        
        super(CVAE, self).__init__()
        
        # Input and cond sanity check
        assert len(input.shape) == 2
        assert len(cond.shape) == 2
        assert cond.shape[0] == input.shape[0]
        
        input_dim = input.shape[1]
        cond_dim = cond.shape[1]

        self.learnable_norm = learnable_norm

        
        # Save some important parameters:
        
        # Save the layer boundaries and the number of layers (both for the dataset). Needed for the layer normalization
        self.layer_boundaries = layer_boundaries_detector
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
        # Logit regularization
        self.alpha = alpha
        
        # the hyperparamters for the loss
        self.sparsity_loss_strength = sparsity_loss
        self.beta = beta
        self.gamma = torch.tensor(gamma)
        
        # needed for the smearing matrix (geometry info)
        self.particle_type = particle_type
        self.dataset = dataset # The number of the dataset 1,2 or 3...
        
        # parameters for layer normalization stability
        self.eps = eps
        
        # Save whether noise layers were used
        self.threshold = threshold
        
        # Save the latent dimension
        self.latent_dim = latent_dim
        
        # Create a logit preprocessing
        self.logit_trafo_in = LogitTransformationVAE(alpha=alpha)
        
        # Create encoder and decoder model as DNNs
        self._set_submodels(input_dim, cond_dim, latent_dim, hidden_sizes)
        
        # Add sigmoid layer
        self.logit_trafo_out = LogitTransformationVAE(alpha=alpha, rev=True)        

              
        # get normalization for normalization layer and to ensure that the incident energy parameter is
        # between 0 and 1:
        self._set_normalizations(input, cond)

        if dataset != 3 and dataset != 2:
            # Create the smearing matrix. It is used in the reco-loss (For DS 2&3 it is to large for the ram)
            self.smearing_matrix = self._get_smearing_matrix(input, cond, smearing_self, smearing_share)
        else:
            self.smearing_matrix = None

    def _set_submodels(self, input_dim, cond_dim, latent_dim, hidden_sizes):
        """Creates the encoder and decoder model as fully connected neural networks."""
        # Create decoder and encoder
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # Add the layers to the encoder
        in_size = input_dim + (cond_dim-self.num_detector_layers) # We do not pass the actual layer energies. They cannot be normalized consistently using only the training set! (Also: Redundant)
            
        for i, hidden_size in enumerate(hidden_sizes):
            self.encoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.encoder.add_module(f"relu{i}", nn.ReLU())
            in_size = hidden_size
        self.encoder.add_module("fc_mu_logvar", nn.Linear(in_size, latent_dim*2))
    
        # add the layers to the decoder
        in_size = latent_dim + (cond_dim-self.num_detector_layers) # We do not pass the actual layer energies. They cannot be normalized consistently using only the training set! (Also: Redundant)
            
        for i, hidden_size in enumerate(reversed(hidden_sizes)):
            self.decoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.decoder.add_module(f"relu{i}", nn.ReLU())
            in_size = hidden_size
            
            
        self.decoder.add_module("fc_out", nn.Linear(in_size, input_dim))
    
    def _set_normalizations(self, data, cond):
        """Initializes the norm layer (Norm to zero mean, unit variance)"""
        # to normalize the incident energy c[:, 0] afterwards. It is not between 0 and 1.
        # The other conditions are allready between 0 and 1 and will not be modified
        # TODO: Problems if test set is more than 15% off
        max_cond_0 = cond[:, [0]].max(axis=0, keepdim=True)[0]
        self.max_cond = torch.cat((max_cond_0, torch.ones(1, cond.shape[1]-1).to(max_cond_0.device)), axis=1)*1.15
        
        # Set the normalization layer operating on the x space (before the actual encoder)
        with torch.no_grad():
            data = self._preprocess_encoding(data, cond, without_norm=True)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        
        
        if self.learnable_norm:
            # Just learn the parameters of the affine transformation
            self.norm_x_in = LearnableNorm([(data.shape[1], )])
            self.norm_x_out = self.norm_x_in
        
        else:
            # Calculate the normalization parameters only once, during the initialization
            initial_scale = 1 / std
            initial_bias = - mean/std
            
            self.norm_x_in = LearnableNorm([(data.shape[1], )], trainable=False, initial_scale=initial_scale, initial_bias=initial_bias)
            self.norm_x_out = self.norm_x_in
        
        return
    
    def _get_smearing_matrix(self, x, c, self_weight=1.0, share_weight=0.0):
        """Computes the smearing matrix that is used in the loss to make neighboring voxels get similar gradients

        Args:
            x (torch.tensor): input data (used to create a hlf file internally. Needed for detector geometry information)
            c (torch.tensor): conditions (used to create a hlf file internally. Needed for detector geometry information)
            self_weight (float, optional): Weight that is sent to the actual voxel (in the loss). Defaults to 1.0.
            share_weight (float, optional):Weight that is sent to the neighboring voxels (in the loss). Defaults to 0.0.
        """
        
        def get_neighboring_indices(index, num_alpha, num_radial):
            
            i, j = divmod(index, num_radial)

            neighbors = []

            left_neighbor = index - num_radial if i > 0 else index + num_radial * (num_alpha - 1)
            if left_neighbor != index:
                
                # links (zyklisch)
                neighbors.append(left_neighbor)
                
                i_l, j_l = divmod(left_neighbor, num_radial)
                
                # unten links
                if j_l > 0:
                    neighbors.append(left_neighbor-1)
                else:
                    neighbor = left_neighbor + num_alpha//2 * num_radial
                    neighbors.append(neighbor % (num_alpha * num_radial))
                    
                # oben links
                if j_l < num_radial - 1:
                    neighbors.append(left_neighbor+1)


            right_neighbor = index + num_radial if i < num_alpha - 1 else index - num_radial * (num_alpha - 1)
            if right_neighbor != index:
                
                # rechts (zyklisch)
                neighbors.append(right_neighbor)
                
                i_r, j_r = divmod(right_neighbor, num_radial)
                
                # unten rechts
                if j_r > 0:
                    neighbors.append(right_neighbor-1)
                else:
                    neighbor = right_neighbor + num_alpha//2 * num_radial
                    neighbors.append(neighbor % (num_alpha * num_radial))
            
                # oben rechts
                if j_r < num_radial - 1:
                    neighbors.append(right_neighbor+1)
                
            # unten
            if j > 0:
                neighbors.append(index - 1)
            else:
                neighbor = index + num_alpha//2 * num_radial
                neighbors.append(neighbor % (num_alpha * num_radial))
            # oben
            if j < num_radial - 1:
                neighbors.append(index + 1)
            
            return neighbors

        hlf_true = data_util.get_hlf(x, c, self.particle_type, self.layer_boundaries, threshold=1.e-10, dataset=self.dataset)

        smearing_matrix = np.zeros((len(hlf_true.showers[0]), len(hlf_true.showers[0])))
        smearing_matrix.shape

        self.num_alphas = []
        self.num_radials = []
        for layer_nr in range(len(self.layer_boundaries)-1):
            
            # needed since we are working with sliced data arrays
            offset = self.layer_boundaries[layer_nr]
            
            # Load the data for the current layer
            reduced_data = hlf_true.showers[0, self.layer_boundaries[layer_nr]:self.layer_boundaries[layer_nr+1]]
            
            # reshape to get number of angles of number of circles
            reduced_data = reduced_data.reshape(int(hlf_true.num_alpha[layer_nr]), -1)
            num_alpha = int(hlf_true.num_alpha[layer_nr])
            num_radial = reduced_data.shape[1]
            
            self.num_alphas.append(num_alpha)
            self.num_radials.append(num_radial)
            
            # use old shape again
            reduced_data = reduced_data.reshape(-1)
            
            # Calculate the actual smearing matrix
            for index, elem in enumerate(reduced_data):
                neighbors = get_neighboring_indices(index, num_alpha, num_radial)

                for neighbor in neighbors:
                    smearing_matrix[index+offset, neighbor+offset] = share_weight
                
                smearing_matrix[index+offset, index+offset] = self_weight

        # hlf_true.DrawSingleShower(smearing_matrix @ hlf_true.showers[0])
        # hlf_true.DrawSingleShower(hlf_true.showers[0])
        
        
        return torch.tensor(smearing_matrix, dtype=torch.get_default_dtype(), device=x.device)
    
    def update_smearing_matrix(self, x, c, self_weight, share_weight):
        
        self.smearing_matrix = self._get_smearing_matrix(x, c, self_weight, share_weight)
        
    def _preprocess_encoding(self, x, c, without_norm=False):
        """First part of the encoder function. Seperated such that it can be used by the
        initialization of the normalization to zero mean and unit variance"""
        

        # Normalize each calo layer to an energy of 1
        x_0_1 = data_util.normalize_layers(x, self.layer_boundaries, c=c, eps=self.eps)
        
        # Needed to ensure numerical stability
        x_0_1 = x_0_1*0.9
            
        # append all extra energy dimensions (the u variables) & Possible other needed conditions
        # We add einc after the normalization since it would results in nans for a single slice, otherwise
        
        # max_cond is not moved with .to() since it is not a parameter
        if self.max_cond.device != x_0_1.device:
            self.max_cond = self.max_cond.to(x.device)
                    
        y_0_1 = torch.cat((x_0_1, (c/self.max_cond)[:, 0:-self.num_detector_layers]), axis=1)
            
        # Go to logit space
        y_logit = self.logit_trafo_in(y_0_1)
        
        # Needed to initialize the norm transformation
        if without_norm:
            return y_logit
        
        else:
            return(self.norm_x_in( (y_logit, ), rev=False)[0][0])
            
    def encode(self, x, c):
        """Takes a point in the dataspace and returns a point in the latent space before sampling.
        Does apply the logit preprocessing and the layer normalization
        noise=True decides, if the noise layer should be used (if it exists)
    
        Output: mu, logvar"""
        
        # Input sanity check
        assert len(x.shape) == 2
        assert len(c.shape) == 2
        assert c.shape[0] == x.shape[0]
        
        # Add noise, normalize layer energies, do logit, normalize to zero mean and unit variance
        x = self._preprocess_encoding(x, c)
        
        # Call the encoder
        mu_logvar = self.encoder(x)

        mu, logvar = mu_logvar[:, :self.latent_dim], mu_logvar[:, self.latent_dim:]
        

        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
            
    def decode(self, latent, c, return_directly_after_decoder=False, train=False):
        """Takes a point in the latent space after sampling and returns a point in the dataspace.
        Output: Reconstructed image in data-space """
        
        # max_cond is not moved with .to() since it is not a parameter
        if self.max_cond.device != c.device:
            self.max_cond = self.max_cond.to(c.device)
        
        # Append the incident energy, the extra dims and possible further conditions
        c_clipped = torch.clamp((c / self.max_cond)[:, 0:-self.num_detector_layers], min=0, max=1)

        # Transform cond into logit space, apply the norm and append in to latent results
        c_logit = self.logit_trafo_in(c_clipped)
        
        # TODO: A norm for the decoder conditions would be a reasonable addition
        # c_prep = self.norm_c( (c_logit, ), rev=False)[0][0]
        c_prep = c_logit	
        
        latent = torch.cat((latent, c_prep), axis=1)
            
        # decode
        x_reco_prep = self.decoder(latent)
        
        if return_directly_after_decoder:
            return x_reco_prep
        
        # Undo normalization step (zero mean, unit variance)
        x_reco_logit = self.norm_x_out( (x_reco_prep, ), rev=True)[0][0]
            
        # Leave the logit space
        x_reco_0_1 = self.logit_trafo_out(x_reco_logit)
        
        # Remove negative energies
        x_reco_0_1[x_reco_0_1<0] = 0
        
        # Revert layer normalization
        x_reco = data_util.unnormalize_layers(x_reco_0_1, c, self.layer_boundaries, eps=self.eps, noise_width=None)
        
        # Threshold the data if a threshold was specified
        if self.threshold is not None:
            x_reco[x_reco < self.threshold] = 0
            
        else:
            # Otherwise the norm before the logit in the reco loss might produce wrong results...
            x_reco[x_reco < 0] = 0
            
        if not train:
            return x_reco
        
        else:
            return x_reco, c
       
    def forward(self, x, c, return_mu_logvar=False):
        """Does the forward pass of the network. Needs the data and the condition. If a noise was specified,
        the model will apply noise to the data and remove it in the logit-space by thresholding.

        Args:
            x (torch.tensor): Input data of dimension (#points, features). The features must contain the energy dimensions as last feature
            c (torch.tensor): Input data of dimension (#points, 1)
            return_mu_logvar (bool): Whether the latent parameters should be passed as well (default=False)

        Returns:
            torch.tensor: reconstruction
        """

        # Encode
        mu, logvar = self.encode(x=x, c=c)
        
        # Sample
        latent = self.reparameterize(mu, logvar)
        
        
        if not return_mu_logvar:
            
            # Decode
            x_reco_shifted = self.decode(latent=latent, c=c, train=False)
            
            return x_reco_shifted
        
        else:
            
            # Decode
            x_reco_shifted, c = self.decode(latent=latent, c=c, train=True)
            
            return x_reco_shifted, c, mu, logvar
    
    def reco_loss(self, x, c, zero_logit=True):
        """Computes the reconstruction loss in the logit space and in the data space"""
        
        # Model forward pass
        x_reco_shifted, c_reco, mu, logvar = self.forward(x=x, c=c, return_mu_logvar=True)
                
        
        # For BCE loss part
        x_0_1      = data_util.normalize_layers(x, self.layer_boundaries, eps=self.eps) * 0.9
        x_reco_0_1 = data_util.normalize_layers(x_reco_shifted, self.layer_boundaries, eps=self.eps) * 0.9
        
        
        # Could also add a possibility to sample here?
        mu_reco_x_0_1 = x_reco_0_1
        
        # For logit loss part
        x_logit = self.logit_trafo_in(x_0_1)
        x_reco_logit = self.logit_trafo_in(mu_reco_x_0_1)
        
        # Compute the losses
        
        # Sparsity loss
        if self.sparsity_loss_strength is not None:
            sparsity_loss = self.sparsity_loss_strength*0.5*nn.functional.mse_loss(smooth_sparsity(x_reco_shifted, 1.e-6), smooth_sparsity(x, 1.e-6))
                
        else:
            sparsity_loss = torch.tensor(0.0, device=x.device)


        # Reconstruction loss
        # Logit loss
        if zero_logit:
            reco_loss_logit = torch.tensor(0.).to(x.device)
            
        else:
            if self.smearing_matrix is not None:
                reco_loss_logit = 0.5*nn.functional.l1_loss(x_reco_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
            else:
                reco_loss_logit = 0.5*nn.functional.l1_loss(x_reco_logit, x_logit, reduction="mean")
        

        # Data BCE loss
        reco_loss_data = self.gamma * 0.5*torch.nn.functional.binary_cross_entropy(x_reco_0_1, x_0_1, reduction="mean")
                

        # KL loss
        KLD = self.beta*torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))


        return reco_loss_logit + reco_loss_data + KLD + sparsity_loss, reco_loss_logit, reco_loss_data, KLD, sparsity_loss


# Building block for the Kernel-VAE
class Block(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super(Block, self).__init__()
        
        self.layers = nn.Sequential()
        
        last_size = input_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.add_module(f"fc{i}", nn.Linear(last_size, hidden_size))
            self.layers.add_module(f"relu{i}", nn.ReLU())
            last_size = hidden_size
            
        self.layers.add_module("fc_out", nn.Linear(last_size, output_dim))
        
    def forward(self, x):
        return self.layers(x)
    

class KernelEncoder(nn.Module):
    
    def __init__(self, input_dim, cond_dim, output_dim, hidden_sizes, hidden_sizes_kernel, kernel_size, kernel_stride, kernel_latent, layer_boundaries):
        super().__init__()
        
        number_detector_layers = len(layer_boundaries)-1

        if number_detector_layers % kernel_stride != 0:
            number_of_blocks = number_detector_layers // kernel_stride + 1
            
        else:
            number_of_blocks = number_detector_layers // kernel_stride
            
            
        self.layer_boundaries = layer_boundaries
            
        self.blocks = nn.ModuleList()

        self.block_boundaries = []
        
        self.cond_dim = cond_dim
        
        for block_index in range(number_of_blocks):
                
                # Calculate the layer boundaries for this block
                start_index = block_index * kernel_stride - (kernel_size-1) // 2
                if start_index < 0:
                    start_index = 0
                    
                end_index = block_index* kernel_stride + (kernel_size-1)//2 + 1
                
                if end_index > number_detector_layers:
                    end_index = number_detector_layers
                    
                    
                self.block_boundaries.append((start_index, end_index))
                    
                # Calculate the input and output dimension for this block
                input_dim_block = layer_boundaries[end_index] - layer_boundaries[start_index] + self.cond_dim
                output_dim_block = kernel_latent
                
                # Create the block
                self.blocks.append(Block(input_dim_block, output_dim_block, hidden_sizes_kernel))


        self.gathering_subnet = nn.Sequential()
        
        # Gather all the individual latent spaces
        last_size = number_of_blocks*kernel_latent + self.cond_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.gathering_subnet.add_module(f"fc{i}", nn.Linear(last_size, hidden_size))
            self.gathering_subnet.add_module(f"relu{i}", nn.ReLU())
            
            last_size = hidden_size
            
        self.gathering_subnet.add_module("fc_out", nn.Linear(last_size, output_dim))
        
    def forward(self, x):
        
        outputs = []

        for i, block in enumerate(self.blocks):
            
            start_index, end_index = self.block_boundaries[i]
            
            block_data = torch.cat([x[..., self.layer_boundaries[start_index]:self.layer_boundaries[end_index]], x[..., -self.cond_dim:]], dim=-1)

            outputs.append(block(block_data))
            
            
        outputs.append(x[..., -self.cond_dim:])           
        gathered = self.gathering_subnet(torch.cat(outputs, dim=-1))
        
        return gathered
      
            
class KernelDecoder(nn.Module):
    
    def __init__(self, input_dim, cond_dim, output_dim, hidden_sizes, hidden_sizes_kernel, kernel_size, kernel_stride, kernel_latent, layer_boundaries):
        super().__init__()
        
        number_detector_layers = len(layer_boundaries)-1

        if number_detector_layers % kernel_stride != 0:
            number_of_blocks = number_detector_layers // kernel_stride + 1
            
        else:
            number_of_blocks = number_detector_layers // kernel_stride
            
            
        self.layer_boundaries = layer_boundaries
            
        self.blocks = nn.ModuleList()

        self.block_boundaries = []
        
        self.cond_dim = cond_dim
        
        self.kernel_latent = kernel_latent
        
        # Go from the true latent space to the individual latent spaces
        self.gathering_subnet = nn.Sequential()
        last_size = input_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.gathering_subnet.add_module(f"fc{i}", nn.Linear(last_size, hidden_size))
            self.gathering_subnet.add_module(f"relu{i}", nn.ReLU())
            last_size = hidden_size
        self.gathering_subnet.add_module("fc_out", nn.Linear(last_size, number_of_blocks*kernel_latent))
        
        
        for block_index in range(number_of_blocks):
                
                # Calculate the layer boundaries for this block
                start_index = block_index * kernel_stride - (kernel_size-1) // 2
                if start_index < 0:
                    start_index = 0
                    
                end_index = block_index * kernel_stride + (kernel_size-1)//2 + 1
                
                if end_index > number_detector_layers:
                    end_index = number_detector_layers
                    
                    
                self.block_boundaries.append((start_index, end_index))
                    
                # Calculate the input and output dimension for this block
                input_dim_block = kernel_latent + self.cond_dim
                output_dim_block = layer_boundaries[end_index] - layer_boundaries[start_index]
                
                # Create the block
                self.blocks.append(Block(input_dim_block, output_dim_block, hidden_sizes_kernel))
        
    def forward(self, x):
        
        
        x = self.gathering_subnet(x)
        
        output = torch.zeros((x.shape[0], self.layer_boundaries[-1]), device=x.device)
        
        for i, block in enumerate(self.blocks):
            
            
            block_data = torch.cat([x[..., i*self.kernel_latent:(i+1)*self.kernel_latent], x[..., -self.cond_dim:]], dim=-1)
            
            start_index, end_index = self.block_boundaries[i]
            
            output[..., self.layer_boundaries[start_index]:self.layer_boundaries[end_index]] += block(block_data)
           
        return output

        
class KernelVAE(CVAE):

    def __init__(self, input, cond, latent_dim, hidden_sizes, hidden_sizes_kernel, layer_boundaries_detector,
                 particle_type="photon", dataset=1, alpha=0.000001, beta=0.00001, gamma=1000,
                 eps=1e-10, smearing_self=1, smearing_share=0, threshold=None,
                 sparsity_loss=None, kernel_size=7, kernel_stride=3, kernel_latent=50, learnable_norm=False):
        
        super().__init__(input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, particle_type, dataset, alpha, beta, gamma, eps, smearing_self, smearing_share, threshold, sparsity_loss, learnable_norm)

        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_latent = kernel_latent
        self.hidden_sizes_kernel = hidden_sizes_kernel
        
        input_dim = input.shape[1]
        cond_dim = cond.shape[1]
        
        self._update_submodels(input_dim, cond_dim, latent_dim, hidden_sizes, hidden_sizes_kernel)    
    
    def _update_submodels(self, input_dim, cond_dim, latent_dim, hidden_sizes, hidden_sizes_kernel):
        # Create decoder and encoder
        
        number_detector_layers = len(self.layer_boundaries)-1
        
        # We do not pass the actual layer energies
        cond_dim = cond_dim - number_detector_layers
        
        self.encoder = KernelEncoder(input_dim=input_dim,
                                     cond_dim=cond_dim,
                                     output_dim=2*latent_dim,
                                     hidden_sizes=hidden_sizes,
                                     hidden_sizes_kernel=self.hidden_sizes_kernel,
                                     kernel_size=self.kernel_size,
                                     kernel_stride=self.kernel_stride,
                                     kernel_latent=self.kernel_latent,
                                     layer_boundaries=self.layer_boundaries)
        
        self.decoder = KernelDecoder(input_dim=latent_dim+cond_dim,
                                     cond_dim=cond_dim,
                                     output_dim=input_dim,
                                     hidden_sizes=hidden_sizes[::-1],
                                     hidden_sizes_kernel=self.hidden_sizes_kernel[::-1],
                                     kernel_size=self.kernel_size,
                                     kernel_stride=self.kernel_stride,
                                     kernel_latent=self.kernel_latent,
                                     layer_boundaries=self.layer_boundaries)


def smooth_sparsity(input, threshold, strength=0.2):
    return (1 / ( 1 + torch.exp( -( (input-threshold) / (strength * threshold))  ) )).mean(axis=1)  
    