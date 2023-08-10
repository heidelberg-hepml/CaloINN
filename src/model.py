import math
import numpy as np
import data_util

import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm

from myBlocks import *
from vblinear import VBLinear
from splines.rational_quadratic import RationalQuadraticSpline

from copy import deepcopy

class Subnet(nn.Module):
    """ This class constructs a subnet for the coupling blocks """

    def __init__(self, num_layers, size_in, size_out, internal_size=None, dropout=0.0,
                 layer_class=nn.Linear, layer_args={}):
        """
            Initializes subnet class.

            Parameters:
            size_in: input size of the subnet
            size: output size of the subnet
            internal_size: hidden size of the subnet. If None, set to 2*size
            dropout: dropout chance of the subnet
        """
        super().__init__()
        if internal_size is None:
            internal_size = size_out * 2
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        self.layer_list = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers -1:
                output_dim = size_out

            self.layer_list.append(layer_class(input_dim, output_dim, **layer_args))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                self.layer_list.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layer_list)

        final_layer_name = str(len(self.layers) - 1)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data.zero_()

    def forward(self, x):
        return self.layers(x)


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
            # if not self.training:
            #     x[:,:-1] = self.norm_logit(x[:,:-1])
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


class NormTransformation(fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, log_cond=False):
        super().__init__(dims_in, dims_c)
        self.log_cond = log_cond

    def forward(self, x, c=None, rev=False, jac=True):
        x, = x
        c, = c
        if self.log_cond:
            c = torch.exp(c)
        if rev:
            z = x/c
            jac = -torch.log(c)
        else:
            z = x*c
            jac = torch.log(c)
        return (z, ), torch.tensor([0.], device=x.device) # jac

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
        self.use_norm = self.params.get("use_norm", False) and not self.params.get("use_extra_dim", False)
        self.pre_subnet = None

        if self.bayesian:
            self.bayesian_layers = []

        self.initialize_normalization(data, cond)
        self.define_model_architecture(self.num_dim)

    def forward(self, x, c, rev=False, jac=True):
        if self.log_cond:
            c_norm = torch.log(c)
        else:
            c_norm = c
        if self.pre_subnet:
            c_norm = self.pre_subnet(c_norm)
        return self.model.forward(x, c_norm, rev=rev, jac=jac)

    def get_constructor_func(self, params):
        """ Returns a function that constructs a subnetwork with the given parameters """
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in params:
            layer_args["prior_prec"] = params["prior_prec"]
        if "std_init" in params:
            layer_args["std_init"] = params["std_init"]
        if "sigma_fixed" in params:
            layer_args["sigma_fixed"] = params["sigma_fixed"]
        def func(x_in, x_out):
            subnet = Subnet(
                    params.get("layers_per_block", 3),
                    x_in, x_out,
                    internal_size = params.get("internal_size"),
                    dropout = params.get("dropout", 0.),
                    layer_class = layer_class,
                    layer_args = layer_args)
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
        if self.use_norm:
            data /= cond
        if self.params.get("logit_transformation", True):
            data = data*(1-2*self.alpha) + self.alpha
            data = torch.logit(data)
        
        elif self.params.get("log_transformation", False):
            data = torch.log(data + self.alpha)
        
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = torch.diag(1 / std)
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
        if self.use_norm:
                nodes.append(ff.Node(
                [nodes[-1].out0],
                NormTransformation,
                {"log_cond": self.log_cond},
                conditions = cond_node,
                name = "norm"
            ))
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
            fm.FixedLinearTransform,
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

    def disenable_map(self):
        for layer in self.bayesian_layers:
            layer.disenable_map()
            
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
        
        # if self.rev:
        #     z = torch.exp(x) - self.alpha
        #     jac = torch.sum( x, dim=1)
        # else:
        #     z = torch.log(x + self.alpha)
        #     jac = - torch.sum( z, dim=1)

        return z
    
    # def forward(self, x):
    #     if self.rev:
    #         z = torch.exp(x) - self.alpha
    #         jac = torch.sum( x, dim=1)
    #     else:
    #         z = torch.log(x + self.alpha)
    #         jac = - torch.sum( z, dim=1)
    #     return z
        
    def __call__(self, x):
        return self.forward(x)
 
class Identity:
    def __init__(self, rev=False, alpha=1.e-6):
        self.alpha = alpha
        self.rev = rev
        
    def forward(self, x):
        return x
    
        
    def __call__(self, x):
        return self.forward(x)
 
class noise_layer(nn.Module):
    def __init__(self, noise_width, layer_boundaries, device, rev):
        super().__init__()
        
        self.noise_width = noise_width
        self.noise_distribution = torch.distributions.Uniform(torch.tensor(0., device=device), torch.tensor(1., device=device))
        self.rev = rev
        
        self.layer_boundaries = layer_boundaries
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
    def forward(self, input, c, noise=None):
        
        if not self.rev:
                                   
            # add noise to the input
            noise = self.noise_distribution.sample(input.shape)*self.noise_width
            input_with_noise = input + noise.view(input.shape)
                        
            return input_with_noise,  noise
               
        else:
            if noise is None:
                input = input - self.noise_width/2
            else:
                input = input - noise
                
            input = torch.where(input<self.noise_width, torch.tensor(0, device=input.device), input)
            
            return input
        
def smooth_sparsity(input, threshold, strength=0.2):
    return (1 / ( 1 + torch.exp( -( (input-threshold) / (strength * threshold))  ) )).mean(axis=1)  

class CVAE(nn.Module):
    def __init__(self, input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, batch_norm=False,
                 particle_type="photon",dataset=1,dropout=0, alpha=1.e-6, beta=1.e-5, gamma=1.e3, learn_gamma=False, 
                 eps=1.e-10, noise_width=None, smearing_self=1.0, smearing_share=0.0, einc_preprocessing="logit",
                 subtract_noise=False, threshold=None, learn_energies=False, sparsity_loss=None, BCE_mode=None):
        
        super(CVAE, self).__init__()
        
        # Input and cond sanity check
        assert len(input.shape) == 2
        assert len(cond.shape) == 2
        assert cond.shape[0] == input.shape[0]
        assert einc_preprocessing in ["logit", "hot_one"]
        
        input_dim = input.shape[1]
        cond_dim = cond.shape[1]
        
        # self.CE_scaling = 1 / torch.std(input, dim=0)
        # self.CE_scaling = self.CE_scaling / torch.mean(self.CE_scaling)
        
        self.log_c_weight = 1
        
        # Save some important parameters:
        
        # Should we learn an energy embedding?
        self.learn_e = learn_energies
        
        # Save the layer boundaries and the number of layers (both for the dataset). Needed for the layer normalization
        self.layer_boundaries = layer_boundaries_detector
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
        # the hyperparamters for the loss
        self.sparsity_loss_strength = sparsity_loss
        self.BCE_mode = BCE_mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.tensor(gamma)
        
        self.learn_gamma = learn_gamma
        if self.learn_gamma:
            with torch.no_grad():
                self.data_sum = 0
                self.logit_sum = 0
        
        # needed for the smearing matrix (geometry info)
        self.particle_type = particle_type
        self.dataset = dataset # The number of the dataset 1,2 or 3...
        
        # parameters for layer normalization stability
        self.eps = eps
        
        # Save whether noise layers were used
        self.noise_width = noise_width
        self.threshold = threshold
        
        # Save the latent dimension
        self.latent_dim = latent_dim
        
        # Initialize the last_noise parameter
        self.last_noise = None
        self.subtract_noise = subtract_noise



        # Now build the network layers:
        
        # Add a noise adding layer
        if noise_width is not None:
            self.noise_layer_in = noise_layer(noise_width, layer_boundaries=self.layer_boundaries, device=input.device, rev=False)
        
        # Save the einc preprocessing type
        self.einc_preprocessing = einc_preprocessing
        if einc_preprocessing == "hot_one":
            self.incident_energies = torch.unique(cond[..., 0])
        
        # Create a logit preprocessing
        self.logit_trafo_in = LogitTransformationVAE(alpha=alpha)
        # self.logit_trafo_in = Identity(alpha=alpha)
        
        # Create encoder and decoder model as DNNs
        self._set_submodels(input_dim, cond_dim, latent_dim, hidden_sizes, dropout, batch_norm=batch_norm)
        
        # Add sigmoid layer
        self.logit_trafo_out = LogitTransformationVAE(alpha=alpha, rev=True)
        # self.logit_trafo_out = Identity(alpha=alpha, rev=True)
        
        # Add a noise removal layer
        if noise_width is not None:
            self.noise_layer_out = noise_layer(noise_width, layer_boundaries=self.layer_boundaries, device=input.device, rev=True)
        

              
        # get normalization for normalization layer and to ensure that the incident energy parameter is
        # between 0 and 1:
        self._set_normalizations(input, cond)
        
        # Create the smearing matrix. It is used in the reco-loss
        self.smearing_matrix = self._get_smearing_matrix(input, cond, smearing_self, smearing_share)

    def _set_submodels(self, input_dim, cond_dim, latent_dim, hidden_sizes, dropout, batch_norm=False):
        # Create decoder and encoder
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # Add the layers to the encoder
        in_size = input_dim + cond_dim-self.num_detector_layers # We do not pass the actual layer energies. They cannot be normalized consistently using only the training set!
        
        # Increase input dim if we use a hot one encoding
        if self.einc_preprocessing == "hot_one":
            in_size = in_size + len(self.incident_energies) - 1
            
        for i, hidden_size in enumerate(hidden_sizes):
            self.encoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.encoder.add_module(f"relu{i}", nn.ReLU())
            if batch_norm:
                self.encoder.add_module(f"batchnorm{i}", torch.nn.BatchNorm1d(num_features=hidden_size))
            self.encoder.add_module(f"dropout{i}", nn.Dropout(p=dropout))
            in_size = hidden_size
        self.encoder.add_module("fc_mu_logvar", nn.Linear(in_size, latent_dim*2))
    
        # add the layers to the decoder
        if not self.learn_e:
            in_size = latent_dim + cond_dim-self.num_detector_layers # We do not pass the actual layer energies. They cannot be normalized consistently using only the training set!
        else:
            in_size = latent_dim + cond_dim-(2*self.num_detector_layers) # Also remove extra dims -> they are already part of the latent space
            
        
        # Increase input dim if we use a hot one encoding
        if self.einc_preprocessing == "hot_one":
            in_size = in_size + len(self.incident_energies) - 1
            
        for i, hidden_size in enumerate(reversed(hidden_sizes)):
            self.decoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.decoder.add_module(f"relu{i}", nn.ReLU())
            if batch_norm:
                self.decoder.add_module(f"batchnorm{i}", torch.nn.BatchNorm1d(num_features=hidden_size))
            self.decoder.add_module(f"dropout{i}", nn.Dropout(p=dropout))
            in_size = hidden_size
            
        # we want to predict the extra dimensions as well
        if self.learn_e:
            input_dim += self.num_detector_layers
            
        self.decoder.add_module("fc_out", nn.Linear(in_size, input_dim))
    
    def _set_normalizations(self, data, cond):
        """Initializes the norm layer (Norm to zero mean, unit variance)"""
        # to normalize the incident energy c[:, 0] afterwards. It is not between 0 and 1.
        # The other conditions are allready between 0 and 1 and will not be modified
        # TODO: Problems if test set is more than 15% off
        max_cond_0 = cond[:, [0]].max(axis=0, keepdim=True)[0]
        self.max_cond = torch.cat((max_cond_0, torch.ones(1, cond.shape[1]-1).to(max_cond_0.device)), axis=1)*1.15
        
        # Set the normalization layer operating on the x space (before the actual encoder)
        data = self._preprocess_encoding(data, cond, without_norm=True)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)      
        self.norm_m_x = torch.diag(1 / std)
        self.norm_b_x = - mean/std
        
        self.norm_x_in = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x, b=self.norm_b_x)
  
        # Find out the slicing boundaries for the norm matrices in the output:
        # (True layer energies and e_inc are removed before the first contact with the norm trafo)
        
        number_of_layers = self.num_detector_layers
        n_extra_dims = cond[..., 1:number_of_layers+1].shape[1]
        n_furhter_conds = cond[..., number_of_layers+2: -number_of_layers].shape[1]       
        print(f"Found {n_furhter_conds} unusual additional conditions during the VAE initialization")
        
        # We do not predict possible further dimensions
        if self.learn_e:
            cut = n_furhter_conds
        
        # The decoder does not predict the extra dimensions and possible further conditions
        else:
            cut = n_furhter_conds+n_extra_dims
            
        if cut != 0:
            self.norm_x_out = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x[:-cut, :-cut], b=self.norm_b_x[:-cut])
        else:
            self.norm_x_out = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x, b=self.norm_b_x)
 
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

                smearing_matrix[index+offset, index+offset] = self_weight
                for neighbor in neighbors:
                    smearing_matrix[index+offset, neighbor+offset] = share_weight
                

        # hlf_true.DrawSingleShower(smearing_matrix @ hlf_true.showers[0])
        # hlf_true.DrawSingleShower(hlf_true.showers[0])
        
        
        return torch.tensor(smearing_matrix, dtype=torch.get_default_dtype(), device=x.device)
    
    def update_smearing_matrix(self, x, c, self_weight, share_weight):
        
        self.smearing_matrix = self._get_smearing_matrix(x, c, self_weight, share_weight)
        
    def _preprocess_encoding(self, x, c, without_norm=False):
        """First part of the encoder function. Seperated such that it can be used by the
        initialization of the normalization to zero mean and unit variance"""
        
        # Adds noise to the data and updates c s.t. the layer normalization will work.
        
        # Add noise -> We must recompute the layer energies
        if self.noise_width is not None:
            x_noise, self.last_noise = self.noise_layer_in(x, c)
            x_noise = data_util.normalize_layers(x_noise, self.layer_boundaries, eps=self.eps)
            
        # Add no noise -> Reuse layer energies
        else:
            x_noise = x
            x_noise = data_util.normalize_layers(x_noise, self.layer_boundaries, c=c, eps=self.eps)
            
        if not self.subtract_noise:
            self.last_noise = None
        
        # Needed to ensure numerical stability
        x_noise = x_noise*0.9
            
        # append all extra energy dimensions (the u variables) & Possible other needed conditions
        # We add einc after the normalization since it would results in nans for a single slice, otherwise
        # TODO: Cross check if results get worse. Since I am only using hot one right now, I cannot check it easily...
        x_noise = torch.cat((x_noise, (c/self.max_cond)[:, 1:-self.num_detector_layers]), axis=1)
            
        # Go to logit space
        x_logit_noise = self.logit_trafo_in(x_noise)
        
        # Needed to initialize the norm transformation
        if without_norm:
            return x_logit_noise
        
        else:
            x_normed = self.norm_x_in( (x_logit_noise, ), rev=False)[0][0]
            
            # Here we append einc just as a logit number
            if self.einc_preprocessing == "logit":
                e_inc_logit = self.logit_trafo_in( c[:, [0]] / self.max_cond[:, [0]] )
                return torch.cat((x_normed, e_inc_logit ), axis=1)
            
            # Here we compute a hot one embedding for e inc
            elif self.einc_preprocessing == "hot_one":
                
                # Compute the one hot encoding
                # append all incident energies to make sure that every index is always the same
                index_tensor = torch.unique(torch.cat((c[:, 0], self.incident_energies)), sorted=True, return_inverse=True)[1]
                
                # Remove the appended list of incident energies
                hot_one_encoding = torch.nn.functional.one_hot(index_tensor)[:-len(self.incident_energies)]
                
                return torch.cat((x_normed, hot_one_encoding), axis=1)
            
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
         
    def _update_c(self, x, c):
        
        number_of_layers = self.num_detector_layers
        
        incident_energy = c[..., [0]]
        extra_dims = x[..., -number_of_layers:]
        x = x[..., :-number_of_layers]
        
        
        # Just as a reference:
        # extra_dims = c[..., 1:number_of_layers+1]
        # layer_energies = c[..., -number_of_layers:]
        # further_conds = c[..., number_of_layers+2: -number_of_layers]
        
        layer_energies = []
        
        e_tot = extra_dims[..., [0]] * incident_energy
        
        for layer in range(self.num_detector_layers-1):
            
            if layer == 0:
                layer_energy = (e_tot) * extra_dims[..., [layer+1]]
                cumsum_previous_layers = layer_energy
                # TODO: removed clone command, might cause bugs
                # cumsum_previous_layers = torch.clone(layer_energy)
            else:
                layer_energy = (e_tot - cumsum_previous_layers) * extra_dims[..., [layer+1]] 
                cumsum_previous_layers += layer_energy
                
            layer_energies.append(layer_energy)
            
        layer_energies.append(e_tot - cumsum_previous_layers)
        
        # print(extra_dims[0, 0], c[0, 1])
        
        c[..., 1:number_of_layers+1] = extra_dims
        c[..., -number_of_layers:] = torch.cat(layer_energies, axis=1)
        return x, c
            
    def decode(self, latent, c, return_directly_after_decoder=False, train=False):
        """Takes a point in the latent space after sampling and returns a point in the dataspace.
        Output: Reconstructed image in data-space """
        
        # Append the extra dims if we did not learn them
        if not self.learn_e:
            c_clipped = torch.clamp((c / self.max_cond)[:, 1:-self.num_detector_layers], min=0, max=1)
            
        # Only append possible further conditions
        else:
            c_clipped = torch.clamp((c / self.max_cond)[:, self.num_detector_layers+2:-self.num_detector_layers], min=0, max=1)
            
        
        # Transform cond into logit space and append to latent results
        c_logit = self.logit_trafo_in(c_clipped)
        latent = torch.cat((latent, c_logit), axis=1)
        
        
        # Compute the one hot encoding if needed
        if self.einc_preprocessing == "hot_one":
            # append all possible incident energies to make sure that every index is always the same
            index_tensor = torch.unique(torch.cat((c[:, 0], self.incident_energies)), sorted=True, return_inverse=True)[1]
            
            # Remove the appended list of incident energies(completely artificial and unneeded)
            hot_one_encoding = torch.nn.functional.one_hot(index_tensor)[:-len(self.incident_energies)]
            
            # Append the correct hot one encoding to the latent space
            latent = torch.cat((latent, hot_one_encoding), axis=1)
            
        # Otherwise append e_inc
        elif self.einc_preprocessing == "logit":
            e_inc = torch.clamp( c[:, [0]] / self.max_cond[:, [0]], min=0, max=1 )
            e_inc_logit = self.logit_trafo_in( e_inc )
            latent = torch.cat((latent, e_inc_logit ), axis=1)
            
                  
        # decode
        x_recon_logit_noise = self.decoder(latent)
        
        if return_directly_after_decoder:
            return x_recon_logit_noise
        
        
        
        # Undo normalization step (zero mean, unit variance)
        x_recon_logit_noise = self.norm_x_out( (x_recon_logit_noise, ), rev=True)[0][0]
            
        # Leave the logit space
        x_recon_noise = self.logit_trafo_out(x_recon_logit_noise)
        
        x_recon_noise[x_recon_noise<0] = 0
        
        if self.learn_e:
            x_recon_noise, c = self._update_c(x_recon_noise, c)
        
        
        # Revert to original normalization -> Enforce layer energies to be correct
        # NOTE: When normalizing to a noise width != 0, I found weird artifacts in the layer energy distribution and bad classifier scores
        # -> Maybe inverstigate...
        # x_recon_noise = data_util.unnormalize_layers(x_recon_noise, c, self.layer_boundaries, eps=self.eps, noise_width=self.noise_width)
        x_recon_noise = data_util.unnormalize_layers(x_recon_noise, c, self.layer_boundaries, eps=self.eps, noise_width=None)
        
        
        # Remove noise by thresholding, if needed
        if self.noise_width is not None:
            if self.last_noise is not None:
                x_recon = self.noise_layer_out(x_recon_noise, c, noise=self.last_noise)
                self.last_noise = None
            else:
                x_recon = self.noise_layer_out(x_recon_noise, c, noise=self.last_noise)
        else:
            x_recon = x_recon_noise
            
        if self.threshold is not None:
            x_recon[x_recon < self.threshold] = 0
        else:
            # Otherwise the norm before the logit in the reco loss might produce wrong results!
            x_recon[x_recon < 0] = 0
            
            
        if self.noise_width is not None:
                        
            # Enforce the total energy to be correct
            # total_energy = c[..., [0]]*c[..., [1]]
            # x_recon_shifted = x_recon * total_energy / (x_recon.sum(axis=1, keepdims=True)+self.eps)
            x_recon_shifted = x_recon
            
            
        else:
            # total_energy = c[..., [0]]*c[..., [1]]
            # x_recon_shifted = x_recon * total_energy / (x_recon.sum(axis=1, keepdims=True)+self.eps)
            x_recon_shifted = x_recon
                    
        if not train:
            return x_recon_shifted
        else:
            return x_recon_shifted, c
       
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
    
    def reco_loss(self, x, c, MAE_logit=True, MAE_data=False, zero_logit=False, zero_data=False):
        """Computes the reconstruction loss in the logit space and in the data space"""
        
        # Model forward pass
        x_reco_shifted, c_reco, mu, logvar = self.forward(x=x, c=c, return_mu_logvar=True)
                
        # For data loss part
        x_dimensionless = x / torch.sqrt(x.mean(axis=1, keepdims=True))
        x_reco_dimensionless = x_reco_shifted / torch.sqrt(x.mean(axis=1, keepdims=True))
        # x_dimensionless = x / torch.sqrt(c[..., [0]])
        # x_reco_dimensionless = x_reco / torch.sqrt(c[..., [0]])
        
        # For BCE loss part
        x_0_1      = data_util.normalize_layers(x, self.layer_boundaries, eps=self.eps) * 0.9
        x_reco_0_1 = data_util.normalize_layers(x_reco_shifted, self.layer_boundaries, eps=self.eps) * 0.9
        
        
        # Could also add a possibility to sample here?
        if self.BCE_mode == "continuous":
            # mu_reco_x_0_1 = self._get_CB_mu(x_reco_0_1)
            mu_reco_x_0_1 = x_reco_0_1
        else:
            mu_reco_x_0_1 = x_reco_0_1
        
        # For logit loss part
        x_logit = self.logit_trafo_in(x_0_1)
        x_reco_logit = self.logit_trafo_in(mu_reco_x_0_1)

        if self.learn_e:
            x_0_1 = torch.cat((x_0_1, c[..., 1:self.num_detector_layers+1]*0.9), axis=1)
            x_reco_0_1 = torch.cat((x_reco_0_1, c_reco[..., 1:self.num_detector_layers+1]*0.9), axis=1)
        
        # Compute the losses
        
        # Sparsity loss
        if self.sparsity_loss_strength is not None:
            if self.noise_width is not None:
                sparsity_loss = self.sparsity_loss_strength*0.5*nn.functional.mse_loss(smooth_sparsity(x_reco_shifted, self.noise_width), smooth_sparsity(x, self.noise_width))
            else:
                sparsity_loss = self.sparsity_loss_strength*0.5*nn.functional.mse_loss(smooth_sparsity(x_reco_shifted, 1.e-6), smooth_sparsity(x, 1.e-6))
                
        else:
            sparsity_loss = torch.tensor(0.0, device=x.device)

        # Reconstruction loss
        # Logit loss
        if not MAE_logit:
            reco_loss_logit = 0.5*nn.functional.mse_loss(x_reco_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
        else:
            reco_loss_logit = 0.5*nn.functional.l1_loss(x_reco_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
        
        # Data MSE loss
        if self.BCE_mode is None:
            if not MAE_data:
                reco_loss_data = self.gamma * 0.5*nn.functional.mse_loss(x_reco_dimensionless @ self.smearing_matrix, x_dimensionless @ self.smearing_matrix, reduction="mean")
            else:
                reco_loss_data = self.gamma * 0.5*nn.functional.l1_loss(x_reco_dimensionless @ self.smearing_matrix, x_dimensionless @ self.smearing_matrix, reduction="mean")
            
            # Not needed for gaussian approach
            log_c = torch.tensor(0.0, device=x.device)
                
        # Data BCE loss
        elif self.BCE_mode == "discrete":
            reco_loss_data = self.gamma * 0.5*torch.nn.functional.binary_cross_entropy(x_reco_0_1, x_0_1, reduction="mean")
            
            # Only needed for the continuous bernoulli approach
            log_c = torch.tensor(0.0, device=x.device)
            
        elif self.BCE_mode == "non_binary":
            # reco_loss_data = self.gamma * cross_entropy(x_reco_0_1, self.CE_scaling * x_0_1, reduction="mean")
            reco_loss_data = self.gamma * cross_entropy(x_reco_0_1, x_0_1, reduction="mean")
            
            # Only needed for the continuous bernoulli approach
            log_c = torch.tensor(0.0, device=x.device)
            
            
        elif self.BCE_mode == "continuous":
            reco_loss_data = self.gamma * 0.5*torch.nn.functional.binary_cross_entropy(x_reco_0_1, x_0_1, reduction="mean")
            
            # Calculates the normalization constant for the continuous bernoulli
            log_c = self.gamma * self.log_c_weight * self._get_CB_log_c(x_reco_0_1, reduction="mean")
            
        else:
            raise ValueError("BCE_mode must be None, discrete or continuous")

        
        if self.training and self.learn_gamma: # Make sure to only track the loss from training, and not from validation!!!
            with torch.no_grad():
                self.data_sum += reco_loss_data
                self.logit_sum += reco_loss_logit
        
        # KL loss
        KLD = self.beta*torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))       

        if zero_logit:
            reco_loss_logit = torch.tensor(0.).to(reco_loss_logit.device)
            
        if zero_data:
            reco_loss_data = torch.tensor(0.).to(reco_loss_logit.device)


        return reco_loss_logit + reco_loss_data + KLD + sparsity_loss - log_c, reco_loss_logit, reco_loss_data, KLD, sparsity_loss, log_c

    def update_gamma(self):
                
        if self.learn_gamma:
            with torch.no_grad():
                self.gamma = self.gamma * (self.logit_sum/self.data_sum)
                self.data_sum = 0
                self.logit_sum = 0
        else:
            print("Updating gamma is disabled!")
        return
    

    def _get_CB_log_c(self, x, eps=1.e-5, reduction="mean"):
        """ Calculates the normalization constant for the continuous bernoulli distribution."""
        
        # The output of the network, cliped to avoid numerical instabilities
        x = torch.clamp(x, 1.e-6, 1.-1.e-6) 
        
        # For values close to the critical point the analytical expression is unstable. Thus we use a taylor expansion around 0.5 for these values.
        mask = (torch.abs(x - 0.5) >= eps)
        anayltical_c = torch.log( (torch.log(1. - x) - torch.log(x)) / (1. - 2. * x) )
        approx_c = torch.log(2. + 2./3.*( 1. - 2. * x)**2 )
        log_c  = torch.where(mask, anayltical_c, approx_c)
        
        if reduction=="sum":
            return log_c.sum()
        elif reduction=="mean":
            return log_c.mean()
        else:
            raise ValueError("reduction must be sum or mean")
    
    def _get_CB_mu(self, x, eps=1.e-5):
        """Calculate the expectation value of the continuous bernoulli distribution."""
        
        x = torch.clamp(x, eps, 1.-eps)
        
        mu = x / (2. * x - 1.) + 1 / (2 * torch.arctanh(1 - 2 * x))
        
        mu = torch.where(x==0, 0.5, mu)
        
        return mu
        
    def update_log_c_weight(self, value):
        self.log_c_weight = value


def cross_entropy(inp, trg, reduction="mean"):
    """Calculates the cross entropy between two distributions."""
    
    # batch_size = inp.shape[0]
    # return torch.nn.functional.cross_entropy(inp.view(-1, 1), trg.view(-1, 1), reduction=reduction)
    
    inp = torch.clamp(inp, 1.e-15, 1)
    
    if reduction == "sum":
        return -torch.sum(trg * torch.log(inp))
    
    elif reduction == "mean":
        return -torch.mean(trg * torch.log(inp))
    
    else:
        raise ValueError("reduction must be sum or mean")