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


class DNN(torch.nn.Module):
    """ NN for vanilla classifier """
    def __init__(self, input_dim, num_layer=2, num_hidden=512, dropout_probability=0.,
                 is_classifier=True):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        if is_classifier:
            all_layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*all_layers)

        self.sigmoid_in_BCE = not is_classifier
        
        if self.sigmoid_in_BCE:
            self.loss_fct = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = torch.nn.BCELoss()
            
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.parameters()))

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_trainable = sum([np.prod(p.size()) for p in model_parameters])
        
        print(f"number of parameters (classifier): {n_trainable}", flush=True)
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def loss(self, *args, **kwargs):
        return self.loss_fct(*args, **kwargs)
    
    
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
        
    def __call__(self, x):
        return self.forward(x)
  
# Create noise layer
class noise_layer(nn.Module):
    def __init__(self, noise_width, rev, logit_space=False, data_space=False, logit_function=None):
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
        
    def forward(self, input, c, energy_dimensions=None):
        
        # Prevent inplace modifications on original data
        input = deepcopy(input)
        c = deepcopy(c)
        
        if self.data_space:
            if not self.rev:
                torch.manual_seed(0)
                noise = self.noise_distribution.sample(input.shape)*self.noise_width
                noise = noise.to(input.device)
                return input + noise.reshape(input.shape)
            else:
                input = torch.where(input<self.noise_width, 0, input)
                return input
            
        else:
            if not self.rev:
                # Rescale the noise such that it is (almost) the same as if it was applied in the dataspace before normalization
                
                size_layer_0, size_layer_1, size_layer_2 = data_util.get_layer_sizes(data_flattened=input)
                l_0 = size_layer_0
                l_01 = size_layer_0 + size_layer_1
                

                # Get the energy dimensions
                u1 = input[:,[-3]]
                u2 = input[:,[-2]]
                u3 = input[:,[-1]]
                e_tot = u1*c
                e0 = u2*e_tot
                e1 = u3*(e_tot-e0)
                e2 = e_tot - e0 -e1
                
                # Reverse the data normalization
                input[..., :l_0] = input[..., :l_0] * e0
                input[..., l_0:l_01] = input[..., l_0:l_01] * e1
                input[..., l_01:] = input[..., l_01:] * e2

                # Sample the noise
                noise_width = torch.ones_like(input[:, :-3])*self.noise_width
                torch.manual_seed(0)
                noise = self.noise_distribution.sample(input[:, :-3].shape)*noise_width
                noise = noise.to(input.device)
                noise = noise.reshape(input[:,:-3].shape)
                
                # Add the noise to the normalization factors
                e0 = e0 + noise[..., :l_0].sum(axis=1, keepdims=True)
                e1 = e1 + noise[..., l_0:l_01].sum(axis=1, keepdims=True)
                e2 = e2 + noise[..., l_01:].sum(axis=1, keepdims=True)

                # Might be used if numerical problems happen. Should not be needed since the noise will
                # regularize anyway!
                eps = 0

                # Renormalize noise and data with new energy factors
                noise[..., :l_0] = noise[..., :l_0] / (e0 + eps) 
                noise[..., l_0:l_01] = noise[..., l_0:l_01] / (e1 + eps)
                noise[..., l_01:] = noise[..., l_01:]  / (e2 + eps)
                
                input[..., :l_0] = input[..., :l_0] / (e0 + eps)
                input[..., l_0:l_01] = input[..., l_0:l_01] / (e1 + eps)
                input[..., l_01:-3] = input[..., l_01:-3] / (e2 + eps)
                
                # TODO: Should I remove this?
                # Update energy dimensions
                u1 = (e0+e1+e2)/c
                u2 = e0/(e0+e1+e2)
                u3 = e1/(e1+e2)
                input[..., -3] = u1[...,0]
                input[..., -2] = u2[...,0]
                input[..., -1] = u3[...,0]

                # Add noise and data, return
                input[:, :-3] = input[:, :-3] + noise.reshape(input[:,:-3].shape)
                return input
            else:
                # TODO: Energy dimensions should also be modified, right???
                # Might be a bug right now!
                assert energy_dimensions is not None, "Need energy dimensions if reverse pass is used"
                assert energy_dimensions.shape[1] == 3, "Must be exactly 3 extra dimensions!"
                
                # get the layer shapes
                size_layer_0, size_layer_1, size_layer_2 = data_util.get_layer_sizes(data_flattened=input)
                l_0 = size_layer_0
                l_01 = size_layer_0 + size_layer_1
                
                # Get the energy dimensions
                u1 = energy_dimensions[:,[0]]
                u2 = energy_dimensions[:,[1]]
                u3 = energy_dimensions[:,[2]]
                e_tot = u1*c
                e0 = u2*e_tot
                e1 = u3*(e_tot-e0)
                e2 = e_tot - e0 -e1
                

                # Sample the noise
                noise_width = torch.ones_like(input[:, :-3])*self.noise_width
                torch.manual_seed(0)
                
                # Approximate the noise here with its expectation value (/2) and do not resample!
                # noise = self.noise_distribution.sample(input[:, :-3].shape)*noise_width
                # noise = noise.to(input.device)
                # noise = noise.reshape(input[:,:-3].shape)
                
                # Add the noise approximation to the normalization factors
                e0 = e0 + noise[..., :l_0].sum(axis=1, keepdims=True) / 2
                e1 = e1 + noise[..., l_0:l_01].sum(axis=1, keepdims=True) / 2
                e2 = e2 + noise[..., l_01:].sum(axis=1, keepdims=True) / 2
                
                normalization = torch.cat((e0.repeat(1, size_layer_0), e1.repeat(1, size_layer_1), e2.repeat(1, size_layer_2)), axis=1).to(input.device)
                
                
                threshold = self.noise_width / normalization
                
                assert threshold.shape == input[:, :-3].shape
                
                if not self.logit_space:
                    input[:, :-3] = torch.where(input[:, :-3]<threshold, 0, input[:, :-3])
                    
                
                else:
                   
                    input[:, :-3] = torch.where(input[:, :-3]<self.logit(threshold), self.logit(0), input[:, :-3])
                    
                return input
        
    
class CVAE(nn.Module):
    def __init__(self, input, cond, latent_dim, hidden_sizes, alpha=1.e-6, beta=1.e-5, gamma=1.e3, noise_width=None):
        super(CVAE, self).__init__()
        
        # Input and cond sanity check
        assert len(input.shape) == 2
        assert len(cond.shape) == 2
        assert cond.shape[0] == input.shape[0]
        assert cond.shape[1] == 4
        
        input_dim = input.shape[1]
        cond_dim = cond.shape[1]

        # Add a noise adding layer
        if noise_width is not None:
            # assert alpha >= noise_width
            print("add noise adding layer")
            self.noise_layer_in = noise_layer(noise_width, rev=False, data_space=False)
        
        # Create a logit preprocessing
        self.logit_trafo_in = LogitTransformationVAE(alpha=alpha)
        
        # Create encoder and decoder model as DNNs
        self.__set_submodels(input_dim, cond_dim, latent_dim, hidden_sizes)
        
        # Add a noise removal layer
        if noise_width is not None:
            assert alpha >= noise_width
            print("add noise removal layer")
            self.noise_layer_out = noise_layer(noise_width, rev=True, data_space=False, logit_function=self.logit_trafo_in)
        
        # Add sigmoid layer
        self.logit_trafo_out = LogitTransformationVAE(alpha=alpha, rev=True)
        
        # the hyperparamters for the reco_loss
        self.beta = beta
        self.gamma = gamma
        
        # Save whether noise layers were used
        self.noise_width = noise_width
        
        # Save the latent dimension
        self.latent_dim = latent_dim
               
        # get normalization for normalization layer and to ensure that the incident energy parameter is
        # between 0 and 1:
        self.__set_normalizations(input, cond)

    def __set_submodels(self, input_dim, cond_dim, latent_dim, hidden_sizes):
        # Create decoder and encoder
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        # Add the layers to the encoder
        in_size = input_dim + cond_dim
        for i, hidden_size in enumerate(hidden_sizes):
            self.encoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.encoder.add_module(f"relu{i}", nn.ReLU())
            in_size = hidden_size
        self.encoder.add_module("fc_mu_logvar", nn.Linear(in_size, latent_dim*2))
    
        # add the layers to the decoder
        in_size = latent_dim + cond_dim
        for i, hidden_size in enumerate(reversed(hidden_sizes)):
            self.decoder.add_module(f"fc{i}", nn.Linear(in_size, hidden_size))
            self.decoder.add_module(f"relu{i}", nn.ReLU())
            in_size = hidden_size
        self.decoder.add_module("fc_out", nn.Linear(in_size, input_dim))
    
    def __set_normalizations(self, data, cond):
        
        # to normalize the incident energy c[:, 0] afterwards. It is not between 0 and 1.
        # The other conditions should be effectively uneffected...
        # TODO: Problems if test set is more than 5% off
        self.max_cond = cond.max(axis=0, keepdim=True)[0]*1.05
        
        # Set the normalization layer operating on the x space (before the actual encoder)
        data = self.__preprocess_encoding(data, cond, without_norm=True)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)      
        self.norm_m_x = torch.diag(1 / std)
        self.norm_b_x = - mean/std
        
        self.norm_x_in = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x, b=self.norm_b_x)
        
        # The decoder does not predict the conditions
        self.norm_x_out = fm.FixedLinearTransform([(data.shape[1], )], M=self.norm_m_x[:-cond.shape[1], :-cond.shape[1]], b=self.norm_b_x[:-cond.shape[1]])
        
        # TODO: Add
        # Set the normalization layer operating on the latent space (before the actual decoder)
        # There one has to normalize the conditions that are appended
        
        return
        
    def __preprocess_encoding(self, x, c, without_norm=False):
        """First part of the encoder function. Seperated such that it can be used by the
        initialization of the normalization to zero mean and unit variance"""
        
        # TODO: Have to add modification of c s.t. normalize layers works properly
        # Add noise, if needed
        if self.noise_width is not None:
            x_noise, c = self.noise_layer_in(x, c)
        else:
            x_noise = x
        
        # Prepares the data by normalizing it and appending the conditions
        x_noise = data_util.normalize_layers(x_noise, c)
        x_noise = torch.cat((x_noise, c/self.max_cond), axis=1)
        
        # Go to logit space
        x_logit_noise = self.logit_trafo_in(x_noise)
        
        if without_norm:
            return x_logit_noise
        
        else:
            return self.norm_x_in( (x_logit_noise, ), rev=False)[0][0]
            
    def encode(self, x, c):
        """Takes a point in the dataspace and returns a point in the latent space before sampling.
        Does apply the logit preprocessing and the layer normalization
        noise=True decides, if the noise layer should be used (if it exists)
    
        Output: mu, logvar"""
        
        # Input sanity check
        assert len(x.shape) == 2
        assert len(c.shape) == 2
        assert c.shape[0] == x.shape[0]
        assert c.shape[1] == 4
        
        # Add noise, normalize layer energies, do logit, normalize to zero mean and unit variance
        x = self.__preprocess_encoding(x, c)
        
        # Call the encoder
        mu_logvar = self.encoder(x)

        mu, logvar = mu_logvar[:, :self.latent_dim], mu_logvar[:, self.latent_dim:]
        

        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def __decode_to_logit(self, latent, c):
        """Takes a point in the latent space after sampling and returns a point in the logit space.
        Output: Reconstructed image in logit-space """
        
        # Transform cond into logit space and append to latent results
        c_logit = self.logit_trafo_in(c / self.max_cond)
        # TODO: Might be useful to call a normalization layer here, too
        latent = torch.cat((latent, c_logit), axis=1)
        
        # decode
        x_recon_logit_noise = self.decoder(latent)
        
        # Undo normalization step (zero mean, unit variance)
        x_recon_logit_noise = self.norm_x_out( (x_recon_logit_noise, ), rev=True)[0][0]
        
        # Remove noise by thresholding, if needed
        if self.noise_width is not None:
            x_recon_logit = self.noise_layer_out(x_recon_logit_noise, c)
        else:
            x_recon_logit = x_recon_logit_noise
            
        return x_recon_logit
            
    def decode(self, latent, c):
        """Takes a point in the latent space after sampling and returns a point in the dataspace.
        Output: Reconstructed image in data-space """
        
        x_recon_logit = self.__decode_to_logit(latent, c)
            
        # Leave the logit space
        x_recon = self.logit_trafo_out(x_recon_logit)
        
        # Revert to original normalization
        x_recon = data_util.unnormalize_layers(x_recon, c)
        
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
    
    def reco_loss(self, x, c):
        """Computes the reconstruction loss in the logit space and in the data space"""
        
        
        # For data part
        x_0_1 = data_util.normalize_layers(x, c)
        
        # For logit loss part
        x_logit = self.logit_trafo_in(x_0_1)
        
        
        
        # Encode
        mu, logvar = self.encode(x=x, c=c)
        
        # Sample
        latent = self.reparameterize(mu, logvar)
        
        # Decode into logit space
        x_recon_logit = self.__decode_to_logit(latent=latent, c=c)
        
        # Leave the logit space
        x_recon = self.logit_trafo_out(x_recon_logit)
        
        # print(x_0_1, x_recon, x_logit, x_recon_logit)
        
        # Compute the losses
        MSE_logit = 0.5*nn.functional.mse_loss(x_recon_logit, x_logit, reduction="mean")
        MSE_data = self.gamma* 0.5*nn.functional.mse_loss(x_recon, x_0_1, reduction="mean")
        KLD = self.beta*torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))
            
        
        return MSE_logit + MSE_data + KLD, MSE_logit, MSE_data, KLD