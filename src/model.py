import math

import torch
import torch.nn as nn
import FrEIA.framework as ff
import FrEIA.modules as fm

from spline_blocks import CubicSplineBlock, RationalQuadraticSplineBlock
from vblinear import VBLinear

class Subnet(nn.Module):
    """ This class constructs a subnet for the coupling blocks
    size_in: input size of the subnet
    size: output size of the subnet
    internal_size: hidden size of the subnet. If None, set to 2*size
    dropout: dropout chance of the subnet
    """

    def __init__(self, num_layers, size_in, size_out, internal_size=None, dropout=0.0,
                 layer_class=nn.Linear, layer_args={}):
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

        final_layer_name = str(len(list(self.layers.modules())) - 2)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name and "logsig2_w" not in name:
                param.data *= 0.02

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
        return (z, ), torch.tensor([1.], device=x.device)

    def output_dims(self, input_dims):
        return input_dims


class CondNormINN(ff.GraphINN):
    def __init__(self, node_list, pre_subnet=None):
        super().__init__(node_list)
        self.pre_subnet = pre_subnet

    def forward(self, x, c, rev=False, jac=True):
        c_norm = torch.log(c)
        if self.pre_subnet is not None:
            c_norm = self.pre_subnet(c_norm)
        return super().forward(x, c_norm, rev=rev, jac=jac)


class CINN(nn.Module):
    """ Class to build, train and evaluate a cINN model """

    def __init__(self, params, device):
        """ Initializes model class with run parameters and the torch device
        instance. """
        super(CINN, self).__init__()
        self.params = params
        self.device = device

        self.verbose = params["verbose"]
        self.parton_norm_m = None
        self.bayesian = params.get("bayesian", False)
        self.alpha = params.get("alpha", 1e-6)
        if self.bayesian:
            self.bayesian_layers = []
        self.losses = {"inn": []}
        if self.bayesian:
            self.losses.update({"kl": [], "total": []})

    def forward(self, x, c, rev=False, jac=True):
        return self.model(x, c, rev=rev, jac=jac)

    def get_constructor_func(self, params):
        """ Returns a function that constructs a subnetwork with the given parameters """
        layer_class = VBLinear if self.bayesian else nn.Linear
        layer_args = {}
        if "prior_prec" in params:
            layer_args["prior_prec"] = params["prior_prec"]
        if "std_init" in params:
            layer_args["std_init"] = params["std_init"]
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
        else:
            raise ValueError(f"Unknown Coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def initialize_normalization(self, data):
        """ Calculates the normalization transformation from the training data. """
        data = torch.log(data + self.alpha)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        self.norm_m = torch.diag(1 / std)
        self.norm_b = - mean/std

    def define_model_architecture(self, in_dim):
        """ Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor """

        self.in_dim = in_dim
        if self.norm_m is None:
            self.norm_m = torch.eye(in_dim)
            self.norm_b = torch.zeros(in_dim)

        nodes = [ff.InputNode(in_dim, name="inp")]
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

        cond_node = ff.ConditionNode(1, name="cond")

        CouplingBlock, block_kwargs = self.get_coupling_block(self.params)
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

        nodes.append(ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)

        self.model = CondNormINN(nodes).to(self.device)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"  Number of parameters: {n_trainable}", flush=True)

    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optim = torch.optim.AdamW(
            self.params_trainable,
            lr = params.get("lr", 0.0002),
            betas = params.get("betas", [0.9, 0.999]),
            eps = params.get("eps", 1e-6),
            weight_decay = params.get("weight_decay", 0.)
        )

        if no_training: return

        self.lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size = params["lr_decay_epochs"],
                gamma = params["lr_decay_factor"],
            )
        elif self.lr_sched_mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                factor = 0.4,
                patience = 50,
                cooldown = 100,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif self.lr_sched_mode == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                params.get("max_lr", params["lr"]*10),
                epochs = params.get("opt_epochs") or params["n_epochs"],
                steps_per_epoch=steps_per_epoch)

    def set_bayesian_std_grad(self, requires_grad):
        for layer in self.bayesian_layers:
            layer.logsig2_w.requires_grad = requires_grad

    def sample_random_state(self):
        return [layer.sample_random_state() for layer in self.bayesian_layers]

    def import_random_state(self, state):
        [layer.import_random_state(s) for layer, s in zip(self.bayesian_layers, state)]

    # def save(self, epoch=""):
    #     """ Save the model, its optimizer, losses, learning rates and the epoch """
    #     if self.angle_enabled and self.angle_cond_mode == "one_hot":
    #         angle_data = { "angle_options": self.angle_options }
    #     else:
    #         angle_data = {}

    #     os.makedirs(self.doc.get_file("model", False), exist_ok=True)
    #     torch.save({"opt": self.optim.state_dict(),
    #                 "net": self.model.state_dict(),
    #                 "losses": self.losses,
    #                 "learning_rates": self.learning_rates,
    #                 "epoch": self.epoch,
    #                 **angle_data}, self.doc.get_file(f"model/model{epoch}", False))

    # def load(self, epoch=""):
    #     """ Load the model, its optimizer, losses, learning rates and the epoch """
    #     name = self.doc.get_file(f"model/model{epoch}", False)
    #     state_dicts = torch.load(name, map_location=self.device)
    #     self.model.load_state_dict(state_dicts["net"])

    #     self.losses = state_dicts.get("losses", {})
    #     if isinstance(self.losses, list):
    #         self.losses = { "inn": self.losses }
    #     self.learning_rates = state_dicts.get("learning_rates", [])
    #     self.epoch = state_dicts.get("epoch", 0)
    #     self.optim.load_state_dict(state_dicts["opt"])
    #     self.model.to(self.device)
    #     if "angle_options" in state_dicts:
    #         self.angle_options = state_dicts["angle_options"]

    def sample(self, num_pts, condition):
        z = torch.normal(0, 1, size=(num_pts*condition.shape[0], self.in_dim), device=self.device)
        c = condition.repeat(num_pts,1)
        x, _ = self.model(z, c, rev=True)
        return x.reshape(num_pts, condition.shape[0], self.in_dim).permute(1,0,2)

    def log_prob(self, x, c):
        z, log_jac_det = self.model(x, c, rev=False)
        log_prob = - 0.5*torch.sum(z**2, 1) + log_jac_det - z.shape[1]/2 * math.log(2*math.pi)
        return log_prob
