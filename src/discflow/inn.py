from FrEIA.framework import *
from FrEIA.modules import *
import numpy as np
import torch
import torch.nn as nn
import time
import math
import os

from one_cycle_lr import OneCycleLR
from util import tqdm_verbose, tqdm_write_verbose, model_class

@model_class
class INN:
    """This class wraps the pytorch model and provides utility functions

    Run parameters:
        n_blocks: number of GLOWCouplingBlocks in the model
        clamping: output clamping of the subnets. Needed for training stability
        dim_cond: dimensions of the conditional input
        internal_size: width of the hidden layers of the subnets in each block
        internal_size_cond: width of the hidden layers of the condition preprocessing
        dim_cond_out: output width of the condition preprocessing
        cond_dropout: dropout probability of the condition preprocessing
        layers_per_block: number of layers the subnets will have in each block
        dropout: dropout probability of the subnets in the coupling blocks
        num_cond_layers: number of layers the condition preprocessing subnet will have
        for more information
        lr: learning rate. This will be decayed if training reaches a plateau
        betas: momentum and squared momentum for ADAM optimizer
        weight_decay: weight decay for regularization
        batch_size: batch size during training. MMD Loss profits from large batch sizes
        masses: Average masses of each of the input partons (after decay)
        eps: epsilon for regularizing the ADAM optimizer forget rates
        masses: the mean masses of the two Bosons
        filename: Name the model should be saved as
        id: Particles to predict (pre decay)
        lam_mmd1: lambda for the first particle's MMD invariant mass loss (IML)
        kernel_name1: kernel to be used for the first particle's IML
        kernel_type1: type of the first kernel
            summed: sum over a list of sigmas
            adaptive: choose the sigma for each batch according to its standard deviation
            standard: use a single fixed sigma
        kernel_width1: sigma(s) for first kernel. Either a single integer of a list of sigmas for summed kernel
        lam_mmd2: lambda for the second particle's IML
        kernel_name2: kernel to be used for the second particle's IML
        kernel_type2: type of the second kernel
        kernel_width2: sigma(s) for second kernel.
        n_epochs: number of epochs to train for
        test_ratio: ratio of test data in the whole training set
        checkpoint_save_interval: the intervals at which te model will be saved during training
        checkpoint_save_overwrite: whether to overwrite old or create new checkpoints during training
        max_val_batches: the maximum number of batches to validate the model on
    """
    def __init__(self, params, data_store, doc):
        self.eval             = False

        self.params           = params
        self.data_store       = data_store
        self.doc              = doc

        self.device           = data_store["device"]
        self.verbose          = params.get("verbose", False)
        self.batch_size       = params.get("batch_size")
        self.epoch            = 0
        self.dim_x            = data_store["dim_x"]

    def get_constructor_func(self, params):
        return lambda x_in, x_out: SubnetConstructor(
                params.get("layers_per_block", 3),
                x_in, x_out,
                internal_size=params.get("internal_size"),
                dropout=params.get("dropout", 0.),
                spectralnorm=params.get("spectralnorm", False))

    def get_coupling_block(self, params):
        constructor_fct = self.get_constructor_func(params)
        permute_soft = params.get("permute_soft")
        coupling_type = params.get("coupling_type", "affine")

        if coupling_type=="affine":
            CouplingBlock = AllInOneBlock
            block_kwargs = {
                            "affine_clamping": params.get("clamping", 5.),
                            "subnet_constructor": constructor_fct,
                            "global_affine_init": 0.92,
                            "permute_soft" : permute_soft
                           }
        elif coupling_type=="cubic":
            CouplingBlock = CubicSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init": params.get("bounds_init", 10),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type=="rational_quadratic":
            CouplingBlock = RationalQuadraticSplineBlock
            block_kwargs = {
                            "num_bins": params.get("num_bins", 10),
                            "subnet_constructor": constructor_fct,
                            "bounds_init":  params.get("bounds_init", 10),
                            "permute_soft" : permute_soft
                           }
        elif coupling_type=="deep_sigmoidal":
            CouplingBlock = DeepSigmoidalBlock
            block_kwargs = {
                            "layers_dsn": params.get("layers_dsn", 2),
                            "subnet_constructor": constructor_fct,
                            "width_dsn":  params.get("width_dsn", 8),
                            "permute_soft" : permute_soft,
                            "mollify" : params.get("mollify", 0.0),
                            "delta" : params.get("delta", 1e-6)
                           }
        else:
            raise ValueError(f"Unknown Coupling block type {coupling_type}")

        return CouplingBlock, block_kwargs

    def define_model_architecture(self):
        """Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor"""

        input_dim = (self.dim_x, 1)
        nodes = [InputNode(*input_dim, name='inp')]

        nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))
        CouplingBlock, block_kwargs = self.get_coupling_block(self.params)

        for i in range(self.params.get("n_blocks", 10)):
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    CouplingBlock,
                    block_kwargs,
                    name = f"block_{i}"
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name='out'))
        self.model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"Number of generator parameters: {n_trainable}", flush=True)

    def get_optimizer(self, params, params_trainable):
        optim_type = params.get("optim_type", "Adam")
        lr = params.get("lr", 0.0002)
        if optim_type == "Adam":
            optim = torch.optim.AdamW(
                params_trainable,
                lr = lr,
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 1e-5)
            )
        elif optim_type == "SGD":
            optim = torch.optim.SGD(
                params_trainable,
                lr = lr,
                momentum = params.get("betas", [0.9, 0.999])[0],
                weight_decay = params.get("weight_decay", 1e-5)
            )
        return optim

    def get_scheduler(self, params, optim, train_loader, steps_per_epoch = None):
        
        lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")
        if lr_sched_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim,
                step_size = params["lr_decay_epochs"],
                gamma = params["lr_decay_factor"],
            )
        elif lr_sched_mode == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                factor = 0.4,
                patience = 10,
                cooldown = 4,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif lr_sched_mode == "one_cycle_lr":           
            if steps_per_epoch is None:     
                steps_per_epoch = len(train_loader)
            print(len(train_loader), steps_per_epoch, flush=True)
            scheduler = OneCycleLR(
                optim,
                params["lr"]*10,
                epochs = params["n_epochs"],
                steps_per_epoch = steps_per_epoch
            )
        return lr_sched_mode, scheduler

    def set_optimizer(self):
        """Set the optimizer for training. ADAM is used with learning rate decay
        on plateau."""
        self.optim = self.get_optimizer(self.params, self.params_trainable)
        self.lr_sched_mode, self.scheduler = self.get_scheduler(self.params, self.optim,
                                                                self.train_loader)

    def initialize_data_loaders(self):
        """Set the model's train and test loader"""
        train_tensor = self.data_store["train_preproc"].to(self.device)
        test_tensor = self.data_store["test_preproc"].to(self.device)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = not self.eval,
        )

        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_tensor),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = not self.eval,
        )

    def validate(self):
        """Validate the model on the test set. Validation will stop if max_val_batches
        is exceeded."""
        loss_tot = 0
        max_val_batches = self.params["max_val_batches"]
        self.model.eval()
        for batch, (x_samps,) in tqdm_verbose(enumerate(self.test_loader), self.verbose,
                                              desc="Validation", leave=False, position=1):
            if batch > max_val_batches:
                break
            x_samps = x_samps.to(self.device)
            gauss_output = self.model(x_samps)
            loss = (torch.mean(gauss_output**2/2)
                   - torch.mean(self.model.log_jacobian(run_forward=False))
                     / gauss_output.shape[1])

            loss_tot += loss.item()
        loss_tot /= min(len(self.test_loader), max_val_batches)
        return loss_tot

    def train(self):
        """Train the model for n_epochs. During training the loss, learning rates
        and the model will be saved in intervals.
        """
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"] = []
        self.data_store["inn_losses"] = []

        start_time = time.time()
        self.model.train()

        for epoch in tqdm_verbose(range(self.params["n_epochs"]), self.verbose,
                                  desc="Epoch", leave=True, position=0):
            epoch_loss = 0

            for batch, (x_samps,) in tqdm_verbose(enumerate(self.train_loader), self.verbose,
                                                  desc="Batch", leave=False, position=1,
                                                  total=len(self.train_loader)):
                x_samps = x_samps.to(self.device)
                self.optim.zero_grad()
                z, jac = self.model(x_samps)

                loss = torch.mean(z**2) / 2 - torch.mean(jac) / z.shape[1]

                #if training diverges, stop
                if not loss < 1e30:
                    print(f"Warning, Loss of {loss.item()} exceeds threshold, " +
                           "skipping back propagation")
                    return

                #save losses
                epoch_loss += loss.item()/len(self.train_loader)

                loss.backward()
                self.optim.step()
                if self.lr_sched_mode == "one_cycle_lr":
                    self.scheduler.step()

            self.epoch += 1

            #save the results of this epoch
            self.data_store["inn_losses"].append(epoch_loss)
            epoch_lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.data_store["learning_rates"].append(epoch_lr)
            tqdm_write_verbose(f"Epoch {self.epoch}: Loss={epoch_loss:.6f}, " +
                               f"LR={epoch_lr:.2e}", self.verbose)

            #handle learning rates
            if self.lr_sched_mode == "reduce_on_plateau":
                self.scheduler.step(self.validate())
            elif self.lr_sched_mode == "step":
                self.scheduler.step()

            #create a backup of the model if it is time
            if isinstance(save_every, list):
                is_save = epoch in save_every
            else:
                is_save = not (epoch % save_every)
            if is_save:
                if save_overwrite:
                    self.save()
                else:
                    self.save(epoch=epoch)

        print("\nTraining complete")
        print(f"--- {time.time() - start_time:.0f} seconds ---")
        print(f"Final train loss: {self.data_store['inn_losses'][-1]:.4f}")

    def predict(self, n_samples):
        gauss_input = torch.randn((n_samples, self.dim_x)).to(self.device)
        events_predict = []
        batch_size = self.params["batch_size"]
        with torch.no_grad():
            for i in tqdm_verbose(range(math.ceil(n_samples / batch_size)), self.verbose,
                                  desc="Generating events", leave=False):
                events_batch = self.model(gauss_input[i * batch_size:(i+1) * batch_size],
                                          rev=True)[0].squeeze()
                events_predict.append(events_batch)
            events_predict = torch.cat(events_predict, dim=0)
        return events_predict

    def compute_latent(self):
        max_latent_batches = self.params.get("max_latent_batches", 10)
        with torch.no_grad():
            self.model.eval()
            rev_data = []
            for batch, (x_samps,) in tqdm_verbose(enumerate(self.test_loader), self.verbose,
                                                  desc="Calculating latent space",
                                                  leave=False, position=1,
                                                  total=max_latent_batches):
                if batch > max_latent_batches:
                    break
                rev_data.append(self.model(x_samps.to(self.device))[0].squeeze())
            return torch.cat(rev_data, dim=0).cpu().detach().numpy()

    def eval_mode(self, evaluate = True):
        """Set the model to eval mode if evaluate == True
        or to train mode if evaluate == False. This is needed so the whole
        dataset is used during testing. (As opposed to dropping the last batch)"""
        self.eval = evaluate
        if self.eval:
            return self.model.eval()
        else:
            return self.model.train()

    def save(self, epoch=""):
        """Save the model, its optimizer and the test/train split, as well as the epoch"""
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "epoch": self.epoch,
                    "preproc": self.preproc}, self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch="", name = None):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        if name is None:
            name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print(f"Warning: Epoch number not provided in save file, setting to {self.epoch}")
        try:
            state_dicts["opt"]["param_groups"][0]["lr"] = self.params.get("lr")
            self.optim.load_state_dict(state_dicts["opt"])
        except ValueError as e:
            print(e)
        try:
            self.preproc.load_state_dict(state_dicts["preproc"])
        except Exception as e:
            print(e, "Warning: Could not load preproc datadata from file")
        self.model.to(self.device)

def _weight_init(m, gain=1.):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.zero_().add_(gain)
        m.bias.data.zero_()

class SubnetConstructor(nn.Module):
    """This class constructs a subnet for the inner parts of the GLOWCouplingBlocks
    as well as the condition preprocessor.
    size_in: input size of the subnet
    size: output size of the subnet
    internal_size: hidden size of the subnet. If None, set to 2*size
    dropout: dropout chance of the subnet
    """

    def __init__(self, num_layers, size_in, size_out, internal_size=None, dropout=0.0, spectralnorm=False,
                 layer_class=nn.Linear):
        super().__init__()
        norm = lambda x: x
        if spectralnorm:
            norm = nn.utils.spectral_norm
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

            self.layer_list.append(layer_class(input_dim, output_dim))

            if n < num_layers - 1:
                if dropout > 0:
                    self.layer_list.append(nn.Dropout(p=dropout))
                self.layer_list.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layer_list)
        self.layers.apply(_weight_init)

        final_layer_name = str(len(list(self.layers.modules())) - 2)
        for name, param in self.layers.named_parameters():
            if name[0] == final_layer_name:
                param.data *= 0.02

    def forward(self, x):
        return self.layers(x)
