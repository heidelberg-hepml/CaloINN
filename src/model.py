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
        # else:
        #     z = torch.log(x + self.alpha)

        return z
    
    # def forward(self, x):
    #     if self.rev:
    #         z = torch.exp(x) - self.alpha
    #     else:
    #         z = torch.log(x + self.alpha)
    #     return z
        
    def __call__(self, x):
        return self.forward(x)
 
    
class IdentityTransformationVAE:
    def __init__(self, rev=False, alpha=1.e-6):
        self.alpha = alpha
        self.rev = rev
        
    def forward(self, x):
        return x
        
    def __call__(self, x):
        return self.forward(x)
 
  
class noise_layer(nn.Module):
    def __init__(self, noise_width, layer_boundaries, rev, logit_space=False, data_space=False, logit_function=None):
        super().__init__()
        
        self.noise_width = noise_width
        self.noise_distribution = torch.distributions.Uniform(torch.tensor(0., device="cpu"), torch.tensor(1., device="cpu"))
        self.rev = rev
        
        assert rev or (not logit_space), "Logit space is only supported for the reverse pass"
        self.logit_space = logit_space
        
        assert (logit_function is not None) or (not logit_space), "Need the actual logit transformation if logit trafos are used"
        self.logit = logit_function
        
        assert (not logit_space) or (not data_space), "Logit space and data space are mutually exclusive!"
        self.data_space = data_space
        
        self.layer_boundaries = layer_boundaries
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
    def forward(self, input, c):
        
        # Prevent inplace modifications on original data
        # TODO: We could prevent clone by just using a different variable name -> less traffic!
        input = torch.clone(input)
        c = torch.clone(c)
        
        if not self.rev:
            # Everything else is not implemented at the moment
            assert self.data_space == True
            assert self.logit_space == False
            
            # add noise to the input
            # #TODO: Prevent the .to comand and create on the device...
            noise = self.noise_distribution.sample(input.shape)*self.noise_width
            noise = noise.to(input.device)
            input = input + noise.view(input.shape)
            
            # rescale the layer energies by the added noise
            incident_energy = c[..., [0]]
            extra_dims = c[..., 1:self.num_detector_layers+1]
            layer_energies = c[..., -self.num_detector_layers:]
            
            # Update the true layer energies
            for layer_index, (layer_start, layer_end) in enumerate(zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])):
                layer_energies[..., [layer_index]] = layer_energies[..., [layer_index]] + noise[..., layer_start:layer_end].sum(axis=1, keepdims=True)
            
            
            # extra dims are not used in the normalize func
            # # Recompute extra dims (Actually not use anywhere!)
            # # Dont need eps here. Regularized by the noise!
            # extra_dims[..., [0]] = layer_energies.sum(axis=1, keepdims=True) / incident_energy
            # for layer_index in range(self.num_detector_layers-1):
            #     extra_dims[..., [layer_index+1]] = layer_energies[..., [layer_index]] / (layer_energies[..., layer_index:]).sum(axis=1, keepdims=True)

            c = torch.cat((incident_energy, extra_dims, c[:, self.num_detector_layers+1:-self.num_detector_layers], layer_energies), axis=1)
            
            return input, c
        
        else:
            # Everything else is not implemented at the moment
            assert self.data_space == False
            assert self.logit_space == True         

            # Get the energy dimensions
            incident_energy = c[..., [0]]
            extra_dims = c[..., 1:self.num_detector_layers+1]
            layer_energies = c[..., -self.num_detector_layers:]
            
            # Approximate the noise here with its expectation value (/2) and do not resample!
            noise_width = torch.ones_like(input, device=input.device)*self.noise_width

            # Add the noise approximation to the normalization factors
            for layer_index, (layer_start, layer_end) in enumerate(zip(self.layer_boundaries[:-1], self.layer_boundaries[1:])):
                layer_energies[..., [layer_index]] = layer_energies[..., [layer_index]] + (noise_width[..., layer_start:layer_end].sum(axis=1, keepdims=True)/2)
                
            
            # Apply the (approximate) layer transformation to the noise threshold 
            # TODO: Repeat is kinda ugly. Maybe better to handle like in the unnormalize func
            all_layer_energies = []
            layer_sizes = self.layer_boundaries[1:] - self.layer_boundaries[:-1]
            for layer_index in range(self.num_detector_layers):
                all_layer_energies = all_layer_energies + [
                    layer_energies[..., [layer_index]].repeat(1, layer_sizes[layer_index])
                    ]    
                
            # all_layer_energies = [energy_layer_1..., energy_layer_2..., ...] 
                
            normalization = torch.cat(all_layer_energies, axis=1).to(input.device)
            threshold = noise_width / normalization
            assert threshold.shape == input.shape

            # Apply the threshold in the logit space by using the monotonicity of the (shifted) logit function
            input = torch.where(input<self.logit(threshold), self.logit(torch.tensor(0, device=input.device)), input)
            
            # extra dims are not used in the unnormalize func
            # # Recompute extra dims needed in order to have a correct unnormalization afterwards
            # # Dont need eps here. Regularized by the noise!
            # extra_dims[..., [0]] = layer_energies.sum(axis=1, keepdims=True) / incident_energy
            # for layer_index in range(self.num_detector_layers-1):
            #     extra_dims[..., [layer_index+1]] = layer_energies[..., [layer_index]] / (layer_energies[..., layer_index:]).sum(axis=1, keepdims=True)

            c_noise = torch.cat((incident_energy, extra_dims, c[:, self.num_detector_layers+1:-self.num_detector_layers], layer_energies), axis=1)
                
            return input, c_noise
        
    
class CVAE(nn.Module):
    def __init__(self, input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, 
                 particle_type="photon",dataset=1,dropout=0, alpha=1.e-6, beta=1.e-5, gamma=1.e3, delta=1., eps=1.e-10, 
                 noise_width=None, smearing_self=1.0, smearing_share=0.0, einc_preprocessing="logit"):
        # TODO: eps is not passed in the normalize and unnormalize funcs. -> Not using the params value
        super(CVAE, self).__init__()
        
        # Input and cond sanity check
        assert len(input.shape) == 2
        assert len(cond.shape) == 2
        assert cond.shape[0] == input.shape[0]
        assert einc_preprocessing in ["logit", "hot_one"]
        
        input_dim = input.shape[1]
        cond_dim = cond.shape[1]

        # Add a noise adding layer
        if noise_width is not None:
            self.noise_layer_in = noise_layer(noise_width, layer_boundaries_detector, rev=False, data_space=True, logit_space=False)
        
        # Save the layer boundaries and the number of layers (both for the dataset). Needed for the layer normalization
        self.layer_boundaries = layer_boundaries_detector
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
        # Save the einc preprocessing type
        self.einc_preprocessing = einc_preprocessing
        if einc_preprocessing == "hot_one":
            self.incident_energies = torch.unique(cond[..., 0])
        
        # Create a logit preprocessing
        self.logit_trafo_in = LogitTransformationVAE(alpha=alpha)
        # self.logit_trafo_in = IdentityTransformationVAE(alpha=alpha)
        
        # Create encoder and decoder model as DNNs
        self._set_submodels(input_dim, cond_dim, latent_dim, hidden_sizes, dropout)
        
        # Add a noise removal layer
        if noise_width is not None:
            self.noise_layer_out = noise_layer(noise_width, layer_boundaries_detector, rev=True, data_space=False, logit_space=True, logit_function=self.logit_trafo_in)
        
        # Add sigmoid layer
        self.logit_trafo_out = LogitTransformationVAE(alpha=alpha, rev=True)
        # self.logit_trafo_out = IdentityTransformationVAE(alpha=alpha, rev=True)
        
        # the hyperparamters for the reco_loss
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # needed for the smearing matrix (geometry info)
        self.particle_type = particle_type
        self.dataset = dataset # The number of the dataset 1,2 or 3...
        
        # parameters for layer normalization stability
        self.eps = eps
        
        # Save whether noise layers were used
        self.noise_width = noise_width
        
        # Save the latent dimension
        self.latent_dim = latent_dim
               
        # get normalization for normalization layer and to ensure that the incident energy parameter is
        # between 0 and 1:
        self._set_normalizations(input, cond)
        
        self.smearing_matrix = self._get_smearing_matrix(input, cond, smearing_self, smearing_share)

    def _set_submodels(self, input_dim, cond_dim, latent_dim, hidden_sizes, dropout):
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
            self.encoder.add_module(f"dropout{i}", nn.Dropout(p=dropout))
            in_size = hidden_size
        self.encoder.add_module("fc_mu_logvar", nn.Linear(in_size, latent_dim*2))
    
        # add the layers to the decoder
        in_size = latent_dim + cond_dim-self.num_detector_layers # We do not pass the actual layer energies. They cannot be normalized consistently using only the training set!
        
        # Increase input dim if we use a hot one encoding
        if self.einc_preprocessing == "hot_one":
            in_size = in_size + len(self.incident_energies) - 1
            
        for i, hidden_size in enumerate(reversed(hidden_sizes)):
            self.decoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.decoder.add_module(f"relu{i}", nn.ReLU())
            self.decoder.add_module(f"dropout{i}", nn.Dropout(p=dropout))
            in_size = hidden_size
        self.decoder.add_module("fc_out", nn.Linear(in_size, input_dim))
    
    def _set_normalizations(self, data, cond):
        
        # to normalize the incident energy c[:, 0] afterwards. It is not between 0 and 1.
        # The other conditions are allready between 0 and 1 and will not be modified
        # TODO: Problems if test set is more than 5% off
        max_cond_0 = cond[:, [0]].max(axis=0, keepdim=True)[0]*1.05
        self.max_cond = torch.cat((max_cond_0, torch.ones(1, cond.shape[1]-1).to(max_cond_0.device)), axis=1)
        
        # Set the normalization layer operating on the x space (before the actual encoder)
        data = self._preprocess_encoding(data, cond, without_norm=True)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)      
        self.norm_m_x = torch.diag(1 / std)
        self.norm_b_x = - mean/std
        
        # # New
        # self.norm_m_x = torch.eye(self.norm_m_x.shape[0], device=self.norm_m_x.device)
        # self.norm_b_x = torch.zeros_like(self.norm_b_x, device=self.norm_b_x.device)
        
        self.norm_x_in = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x, b=self.norm_b_x)
        
        if self.einc_preprocessing == "logit":
            # The decoder does not predict the conditions (-(len(c)-num_layers) instead of -len(c) since we did not pass the last true layer energies to the network)
            self.norm_x_out = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x[:-(cond.shape[1]-self.num_detector_layers), 
                                                                                        :-(cond.shape[1]-self.num_detector_layers)], 
                                                    b=self.norm_b_x[:-(cond.shape[1]-self.num_detector_layers)])
        
        elif self.einc_preprocessing == "hot_one":
            # In this case we must predict relative to logit one dimension more (Einc was allready removed in encode!)
            self.norm_x_out = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x[:-(cond.shape[1]-self.num_detector_layers-1), 
                                                                            :-(cond.shape[1]-self.num_detector_layers-1)], 
                                        b=self.norm_b_x[:-(cond.shape[1]-self.num_detector_layers-1)])
        
        # TODO: Add
        # Set the normalization layer operating on the latent space (before the actual decoder)
        # There one has to normalize the conditions that are appended
        
        return
     
    def _get_smearing_matrix(self, x, c, self_weight=1.0, share_weight=0.0):
        
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

        # TODO: Adjust the particle type
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
        # This c_noise values will only used for the layer normalization, not for the training!
        if self.noise_width is not None:
            x_noise, c_noise = self.noise_layer_in(x, c)
        else:
            x_noise = x
            c_noise = c
        
        x_noise = data_util.normalize_layers(x_noise, c_noise, self.layer_boundaries, eps=self.eps)
        
        if self.einc_preprocessing == "logit":
            # Prepares the data by normalizing it and appending the conditions
            x_noise = torch.cat((x_noise, (c/self.max_cond)[:, :-self.num_detector_layers]), axis=1) # Normalize c and dont append the true layer energies!
            
            # Go to logit space
            x_logit_noise = self.logit_trafo_in(x_noise)
        
        elif self.einc_preprocessing == "hot_one":
            # Difference: Here we also do not append einc (firt index)
            x_noise = torch.cat((x_noise, (c/self.max_cond)[:, 1:-self.num_detector_layers]), axis=1)
            
            # Go to logit space
            x_logit_noise = self.logit_trafo_in(x_noise)
        
        if without_norm:
            return x_logit_noise
        
        else:
            if self.einc_preprocessing == "logit":
                return self.norm_x_in( (x_logit_noise, ), rev=False)[0][0]
            
            elif self.einc_preprocessing == "hot_one":
                
                # Compute the one hot encoding
                # append all incident energies to make sure that every index is always the same
                index_tensor = torch.unique(torch.cat((c[:, 0], self.incident_energies)), sorted=True, return_inverse=True)[1]
                
                # Remove the appended list of incident energies
                hot_one_encoding = torch.nn.functional.one_hot(index_tensor)[:-len(self.incident_energies)]
                
                return torch.cat((self.norm_x_in( (x_logit_noise, ), rev=False)[0][0], hot_one_encoding), axis=1)
            
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

    def _decode_to_logit(self, latent, c):
        """Takes a point in the latent space after sampling and returns a point in the logit space.
        Output: Reconstructed image in logit-space and possibly inflated conditions"""
        
        if self.einc_preprocessing == "logit":
            # Normalize c, clip it (needed for stability in generation) and dont append the true layer energies!
            c_clipped = torch.clamp((c / self.max_cond)[:, :-self.num_detector_layers], min=0, max=1)
            
        elif self.einc_preprocessing == "hot_one":
            c_clipped = torch.clamp((c / self.max_cond)[:, 1:-self.num_detector_layers], min=0, max=1)
            
        
        # Transform cond into logit space and append to latent results
        c_logit = self.logit_trafo_in(c_clipped)
        # TODO: Might be useful to call a normalization layer here, too
        latent = torch.cat((latent, c_logit), axis=1)
        
        if self.einc_preprocessing == "hot_one":
            # Compute the one hot encoding
            # append all incident energies to make sure that every index is always the same
            index_tensor = torch.unique(torch.cat((c[:, 0], self.incident_energies)), sorted=True, return_inverse=True)[1]
            
            # Remove the appended list of incident energies
            hot_one_encoding = torch.nn.functional.one_hot(index_tensor)[:-len(self.incident_energies)]
            
            latent = torch.cat((latent, hot_one_encoding), axis=1)
            
            
        
        # decode
        x_recon_logit_noise = self.decoder(latent)
        
        # Undo normalization step (zero mean, unit variance)
        x_recon_logit_noise = self.norm_x_out( (x_recon_logit_noise, ), rev=True)[0][0]
        
        # Remove noise by thresholding, if needed
        if self.noise_width is not None:
            x_recon_logit, c_noise = self.noise_layer_out(x_recon_logit_noise, c)
        else:
            x_recon_logit = x_recon_logit_noise
            c_noise = c
            
        return x_recon_logit, c_noise
            
    def decode(self, latent, c):
        """Takes a point in the latent space after sampling and returns a point in the dataspace.
        Output: Reconstructed image in data-space """
        
        x_recon_logit, c_noise = self._decode_to_logit(latent, c)
            
        # Leave the logit space
        x_recon = self.logit_trafo_out(x_recon_logit)
        
        # Revert to original normalization
        x_recon = data_util.unnormalize_layers(x_recon, c_noise, self.layer_boundaries, eps=self.eps)
        
        return x_recon
        
    def forward(self, x, c):
        """Does the forward pass of the network. Needs the data and the condition. If a noise was specified,
        the model will apply noise to the data and remove it in the logit-space by thresholding.

        Args:
            x (torch.tensor): Input data of dimension (#points, features). The features must contain the energy dimensions as last feature
            c (torch.tensor): Input data of dimension (#points, 1)
            noise (bool, optional): Whether noise should be added

        Returns:
            torch.tensor: reconstructed data
        """

        # Encode
        mu, logvar = self.encode(x=x, c=c)
        
        # Sample
        latent = self.reparameterize(mu, logvar)
        
        # Decode
        return self.decode(latent=latent, c=c)
    
    def reco_loss(self, x, c, MAE_logit=True, MAE_data=False, zero_logit=False, zero_data=False):
        """Computes the reconstruction loss in the logit space and in the data space"""
        
        # Encode
        mu, logvar = self.encode(x=x, c=c)
        
        # Sample
        latent = self.reparameterize(mu, logvar)
        
        # Decode into logit space
        x_recon_logit, c_noise = self._decode_to_logit(latent=latent, c=c)
        
        # Leave the logit space
        x_recon_0_1 = self.logit_trafo_out(x_recon_logit)
        
        
        # Ground truth preprocessing:
        # For data part (Use c_noise here for better agreement)
        x_0_1 = data_util.normalize_layers(x, c_noise, self.layer_boundaries, eps=self.eps)
        
        # For logit loss part
        x_logit = self.logit_trafo_in(x_0_1)
        
        
        # print()
        # print("recos & originals")
        # print(x_0_1, x_logit, latent, x_recon_logit, x_recon_0_1)
        
        # Compute the losses
        
        # incident_energy = c[..., [0]]
        
        # gamma = self.gamma * torch.sqrt(4.2e1 / incident_energy)
        
        # gamma =  torch.clamp(gamma, min=self.gamma, max=self.gamma*1.e1)
        
        if not MAE_logit:
            MSE_logit = self.delta * 0.5*nn.functional.mse_loss(x_recon_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
        else:
            MSE_logit = self.delta * 0.5*nn.functional.l1_loss(x_recon_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
            
        # if not MAE_data:
        #     MSE_data = 0.5*nn.functional.mse_loss(gamma* x_recon_0_1 @ self.smearing_matrix, gamma* x_0_1 @ self.smearing_matrix, reduction="mean") / x_recon_0_1.shape[0]
        # else:
        #     MSE_data = 0.5*nn.functional.l1_loss(gamma * x_recon_0_1 @ self.smearing_matrix, gamma* x_0_1 @ self.smearing_matrix, reduction="mean") / x_recon_0_1.shape[0]
            
        if not MAE_data:
            MSE_data = self.gamma * 0.5*nn.functional.mse_loss(x_recon_0_1 @ self.smearing_matrix, x_0_1 @ self.smearing_matrix, reduction="mean")
        else:
            MSE_data = self.gamma * 0.5*nn.functional.l1_loss(x_recon_0_1 @ self.smearing_matrix, x_0_1 @ self.smearing_matrix, reduction="mean")
        
        KLD = self.beta*torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))       

        if zero_logit:
            MSE_logit = torch.tensor(0).to(MSE_logit.device)
            
        if zero_data:
            MSE_data = torch.tensor(0).to(MSE_logit.device)
            
        # print()
        # print("losses")
        # print(MSE_logit, MSE_data, KLD)
            
        
        return MSE_logit + MSE_data + KLD, MSE_logit, MSE_data, KLD


class cyclic_padding(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, tensor):
        # shifted mirror padding for lower boundary
        bottom = torch.roll(torch.flip(tensor[:, :, :self.padding], dims=[2]), (tensor.shape[3] // 2), dims=[3])
        
        # TODO: could be faster by just allocating once in __init__()!
        top = torch.zeros_like(bottom, device=tensor.device)
        
        tensor = torch.cat((bottom, tensor, top), dim=2)

        # Cyclic padding on right and left boundary
        if self.padding>0:
            left = tensor[:, :, :, -self.padding:]
        else:
            left = tensor[:, :, :, :0]
        right = tensor[:, :, :, :self.padding]
        tensor = torch.cat((left, tensor, right), dim=3)
        
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(padding={self.padding})'

# Outdated!
class CCVAE(CVAE):
    def __init__(self, input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, kernel_size, channel_list, padding,
                 particle_type="photon", dataset=1, dropout=0, alpha=0.000001, beta=0.00001, gamma=1000, eps=1e-10, noise_width=None,
                 smearing_self=1.0, smearing_share=0.0):
                
        cond_dim = cond.shape[1]
        input_dim = input.shape[1]
        
        # Initialize CVAE
        super().__init__(input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, particle_type, dataset, 
                         dropout, alpha, beta, gamma, eps, noise_width, smearing_self, smearing_share)
        
        # Unroll the data and seperate the detector layers (only take the first 10 points for speedup just need the shapes
        # of the other axes)
        x = self._preprocess_encoding(input, cond)
        unrolled_data = self._unroll_data(x[0:10].to("cpu"))
        
        # Add the small convolutional networks
        self._add_convolutional_overhead(unrolled_data, kernel_size, channel_list, padding)
        
        # print(self.convolutional_heads_encode)
        # print(self.convolutional_heads_decode)
        
        convoluted_data = self._conv_encode(unrolled_data)
        
        self.conv_shapes = []
        
        for conv_output in convoluted_data[:-1]:
            self.conv_shapes.append(conv_output.shape[1:])
                
        x = torch.cat([x.view(x.shape[0], -1) for x in convoluted_data], axis=1)
        
        self.encoder[0] = nn.Linear(x.shape[1], hidden_sizes[0]) 
        # TODO: Will only work as long as we keep the number of conditions fixed in the small fc network
        # self.decoder[-1] = nn.Linear(hidden_sizes[0], x.shape[1]-(cond_dim-self.num_detector_layers)*channel_list[-1])
        self.decoder[-1] = nn.Linear(hidden_sizes[0], input_dim) 
                   
    def _unroll_data(self, data):
        """Unrolls and slices the input data such that the conv operations can use the data"""
        
        unrolled_data_list = []
        for layer_nr in range(self.num_detector_layers):
            
            # Load the data for the current layer
            reduced_data = data[..., self.layer_boundaries[layer_nr]:self.layer_boundaries[layer_nr+1]]
            
            # reshape the data
            if self.num_alphas[layer_nr] > 1:
                unrolled_data_list.append(reduced_data.view(-1, self.num_alphas[layer_nr], self.num_radials[layer_nr]))
            else:
                unrolled_data_list.append(reduced_data)
                
        # append the conditions as another layer:
        unrolled_data_list.append(data[..., self.layer_boundaries[layer_nr+1]:])
            
        return unrolled_data_list
    
    def _add_convolutional_overhead(self, input, kernel_size, channel_list, padding):
        """Adds convoultional networks to the beginning and the end of the model"""
        
        # Add the conv nets for the encoder
        convolutional_heads_encode = []
        
        for layer_data in input:
            
            input_channels = 1
            convolutional_heads_encode.append(nn.Sequential())
            
            # Add a convoutional net if we a 2 dimensional data (3 because of batch dim)
            if len(layer_data.shape) == 3:
                for i, num_channels in enumerate(channel_list):
                    convolutional_heads_encode[-1].add_module(f"{padding}-padding_{i}", cyclic_padding(padding))
                    convolutional_heads_encode[-1].add_module(f"conv_{i}", nn.Conv2d(input_channels, num_channels, kernel_size=kernel_size, stride=1, padding=0))
                    convolutional_heads_encode[-1].add_module(f"relu_{i}", nn.ReLU())
                    
                    input_channels = num_channels
            
            # Add a FC net if we a 1 dimensional data (2 because of batch dim)
            elif len(layer_data.shape) == 2:
                num_channels_prev = 1
                for i, num_channels in enumerate(channel_list):
                    in_dim = layer_data.shape[1]
                    convolutional_heads_encode[-1].add_module(f"fc_{i}", nn.Linear(in_dim*num_channels_prev, in_dim*num_channels))
                    convolutional_heads_encode[-1].add_module(f"relu_{i}", nn.ReLU())
                    num_channels_prev = num_channels
                    
            # In this case something went wrong. Should not happen...
            else:
                raise RuntimeError("Invalid shape for layer data array!")
        
        self.convolutional_heads_encode = nn.ModuleList(convolutional_heads_encode)
        
        
        # Add the conv nets for the decoder
        convolutional_heads_decode = []
        
        for layer_data in input:
            
            input_channels = channel_list[-1]
            convolutional_heads_decode.append(nn.Sequential())
            
            # Add a convoutional net if we a 2 dimensional data (3 because of batch dim)
            if len(layer_data.shape) == 3:
                for i, num_channels in enumerate(channel_list[1::-1]):
                    convolutional_heads_decode[-1].add_module(f"{padding}-padding_{i}", cyclic_padding(kernel_size - 1 - padding))
                    # TODO: Think about the transpose conv again...
                    convolutional_heads_decode[-1].add_module(f"tranpose_conv_{i}", nn.ConvTranspose2d(input_channels, num_channels, kernel_size=kernel_size, stride=1, padding=kernel_size-1))
                    convolutional_heads_decode[-1].add_module(f"relu_{i}", nn.ReLU())
                    
                    input_channels = num_channels
                    
                convolutional_heads_decode[-1].add_module(f"{padding}-padding_{i+1}", cyclic_padding(kernel_size - 1 - padding))
                # TODO: Think about the transpose conv again...
                convolutional_heads_decode[-1].add_module(f"tranpose_conv_out", nn.ConvTranspose2d(input_channels, 1, kernel_size=kernel_size, stride=1, padding=kernel_size-1))
            
            
            # Add a FC net if we a 1 dimensional data (2 because of batch dim)
            elif len(layer_data.shape) == 2:
                    
                num_channels_prev = channel_list[-1]
                for i, num_channels in enumerate(channel_list[1::-1]):
                    in_dim = layer_data.shape[1]
                    convolutional_heads_decode[-1].add_module(f"fc_{i}", nn.Linear(in_dim*num_channels_prev, in_dim*num_channels))
                    convolutional_heads_decode[-1].add_module(f"relu_{i}", nn.ReLU())
                    num_channels_prev = num_channels
                    
                convolutional_heads_decode[-1].add_module(f"fc_out", nn.Linear(in_dim*num_channels_prev, in_dim*1))
                
            # In this case something went wrong. Should not happen...
            else:
                raise RuntimeError("Invalid shape for layer data array!")
            
        
        self.convolutional_heads_decode = nn.ModuleList(convolutional_heads_decode)
    
    def _conv_encode(self, unrolled_data):
        """"Encoder pass through convolutional networks"""
        
        output = []
        
        for layer_data, subnet in zip(unrolled_data, self.convolutional_heads_encode):
            
            if len(layer_data.shape) == 3:
                x = subnet(layer_data.unsqueeze(1))
            else:
                x = subnet(layer_data)
            
                
            output.append(x)
        
        return output
    
    def _conv_decode(self, unrolled_data):
        """"Decoder pass through convolutional networks"""
        
        output = []
        
        for layer_data, subnet in zip(unrolled_data, self.convolutional_heads_decode):
            
            if len(layer_data.shape) == 3:
                x = subnet(layer_data.unsqueeze(1))
            else:
                x = subnet(layer_data)
            
                
            output.append(x)
        
        return output
    
    def encode(self, x, c):
        """Upates this step to include the call of the convolutions. Rest: compare base function"""
        
        # Input sanity check
        assert len(x.shape) == 2
        assert len(c.shape) == 2
        assert c.shape[0] == x.shape[0]
        
        # Add noise, normalize layer energies, do logit, normalize to zero mean and unit variance
        x = self._preprocess_encoding(x, c)

        # 1) Reshape x into matrix shape "data.reshape(num_alpha, num_radial)"
        unrolled_data = self._unroll_data(x)

        # 2) Put x into encoder conv overhead
        convoluted_data = self._conv_encode(unrolled_data)
        
        # 3) concatenate everything back together
               
        x = torch.cat([x.view(x.shape[0], -1) for x in convoluted_data], axis=1)      
        
        # Call the encoder
        mu_logvar = self.encoder(x)

        mu, logvar = mu_logvar[:, :self.latent_dim], mu_logvar[:, self.latent_dim:]
        

        
        return mu, logvar     
        
    def _decode_to_logit(self, latent, c):
        """Upates this step to include the call of the convolutions. Rest: compare base function"""      
        
        # Normalize c, clip it (needed for stability in generation) and dont append the true layer energies!
        c_clipped = torch.clamp((c / self.max_cond)[:, :-self.num_detector_layers], min=0, max=1)
        
        # Transform cond into logit space and append to latent results
        c_logit = self.logit_trafo_in(c_clipped)
        # TODO: Might be useful to call a normalization layer here, too
        latent = torch.cat((latent, c_logit), axis=1)
        
        # decode
        x_recon_logit_noise = self.decoder(latent)
        
        
        # # TODO:
        # # 1) Slice output to the layers
        # x_unrolled = []
        # start = 0
        # for conv_shape in self.conv_shapes:
        #     end = start + np.product(conv_shape)
        #     x_unrolled.append(x_recon_logit_noise[:, start:end].view(-1, *conv_shape))
        #     start = end
            
        # # 2) Put in decoder conv overhead
        # x_up_convoluted = self._conv_decode(x_unrolled)

        # # 3) Reshape and put together
        # x_recon_logit_noise = torch.cat([x.view(x.shape[0], -1) for x in x_up_convoluted], axis=1)
        
        # Undo normalization step (zero mean, unit variance)
        x_recon_logit_noise = self.norm_x_out( (x_recon_logit_noise, ), rev=True)[0][0]
        
        
        
        # Remove noise by thresholding, if needed
        if self.noise_width is not None:
            x_recon_logit, c_noise = self.noise_layer_out(x_recon_logit_noise, c)
        else:
            x_recon_logit = x_recon_logit_noise
            c_noise = c
            
        return x_recon_logit, c_noise
    
# Outdated!
class CAE(CVAE):
    def __init__(self, input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, particle_type="photon", dataset=1, dropout=0, alpha=0.000001, beta=0.00001, gamma=1000, eps=1e-10, noise_width=None, smearing_self=1, smearing_share=0):
        super().__init__(input, cond, latent_dim, hidden_sizes, layer_boundaries_detector, particle_type, dataset, dropout, alpha, beta, gamma, eps, noise_width, smearing_self, smearing_share)
        
        self.encoder[-1] = nn.Linear(hidden_sizes[-1], latent_dim)
    
    def encode(self, x, c):
        """Takes a point in the dataspace and returns a point in the latent space before sampling.
        Does apply the logit preprocessing and the layer normalization
        noise=True decides, if the noise layer should be used (if it exists)
    
        Output: latent"""
        
        # Input sanity check
        assert len(x.shape) == 2
        assert len(c.shape) == 2
        assert c.shape[0] == x.shape[0]
        
        # Add noise, normalize layer energies, do logit, normalize to zero mean and unit variance
        x = self._preprocess_encoding(x, c)
        
        # Call the encoder
        latent = self.encoder(x)
        
        return latent
    
    def forward(self, x, c):
        """Does the forward pass of the network. Needs the data and the condition. If a noise was specified,
        the model will apply noise to the data and remove it in the logit-space by thresholding.

        Args:
            x (torch.tensor): Input data of dimension (#points, features). The features must contain the energy dimensions as last feature
            c (torch.tensor): Input data of dimension (#points, 1)
            noise (bool, optional): Whether noise should be added

        Returns:
            torch.tensor: reconstructed data
        """

        # Encode
        latent = self.encode(x=x, c=c)
        
        
        # Decode
        return self.decode(latent=latent, c=c)
    
    def reco_loss(self, x, c, MAE_logit=True, MAE_data=False, zero_logit=False, zero_data=False):
        """Computes the reconstruction loss in the logit space and in the data space"""
        
        
        # For data part
        x_0_1 = data_util.normalize_layers(x, c, self.layer_boundaries, eps=self.eps)
        
        # For logit loss part
        x_logit = self.logit_trafo_in(x_0_1)
        
        # Encode
        latent = self.encode(x=x, c=c)
        
        # Decode into logit space
        x_recon_logit = self._decode_to_logit(latent=latent, c=c)
        
        # Leave the logit space
        x_recon_0_1 = self.logit_trafo_out(x_recon_logit)
        
        # print()
        # print("recos & originals")
        # print(x_0_1, x_logit, latent, x_recon_logit, x_recon_0_1)
        
        # Compute the losses
        
        if not MAE_logit:
            MSE_logit = 0.5*nn.functional.mse_loss(x_recon_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
        else:
            MSE_logit = 0.5*nn.functional.l1_loss(x_recon_logit @ self.smearing_matrix, x_logit @ self.smearing_matrix, reduction="mean")
            
        if not MAE_data:
            MSE_data = self.gamma* 0.5*nn.functional.mse_loss(x_recon_0_1 @ self.smearing_matrix, x_0_1 @ self.smearing_matrix, reduction="mean")
        else:
            MSE_data = self.gamma* 0.5*nn.functional.l1_loss(x_recon_0_1 @ self.smearing_matrix, x_0_1 @ self.smearing_matrix, reduction="mean") 

        if zero_logit:
            MSE_logit = torch.tensor(0).to(MSE_logit.device)
            
        if zero_data:
            MSE_data = torch.tensor(0).to(MSE_logit.device)
            
        # print()
        # print("losses")
        # print(MSE_logit, MSE_data, KLD)
            
        
        return MSE_logit + MSE_data, MSE_logit, MSE_data, torch.tensor(0)