from FrEIA.framework import *
from FrEIA.modules import *
import numpy as np
import torch
import torch.nn as nn
import time
from math import ceil, floor
import os
from inn import INN
from one_cycle_lr import OneCycleLR
from util import tqdm_verbose, tqdm_write_verbose, model_class, eval_observables_list
from tqdm import tqdm
import preprocessing
from gradient_penalty import Discriminator_Regularizer
from losses import *

@model_class
class iGAN(INN):
    """This class wraps the pytorch model and provides utility functions

    Run parameters:
        See INN class for additional parameters

    """
    def __init__(self, params, data_store, doc):
        super().__init__(params, data_store, doc)

        params["dim_x"] = self.dim_x
        gen_iter = params.get("gen_batch_periter", 1)
        disc_iter = params.get("disc_batch_periter", 1)
        params["batch_ratio"] = gen_iter/(gen_iter + disc_iter)
        self.obs_converter = eval_observables_list(
            params.get("disc_observables", params["input_observables"]))
        params["dim_disc"] = len(self.obs_converter)
        self.obs_converter_back = eval_observables_list(
            params["input_observables"])
        self.BCE = nn.BCEWithLogitsLoss()
        self.provided_values = [(eval_observables_list(obs_expr), value)
                                for obs_expr, value in self.params.get("provided_values", [])]

    def define_model_architecture(self):
        """Create a ReversibleGraphNet model based on the settings, using
        SubnetConstructor as the subnet constructor"""
        if not self.params.get("noINN", False):
            super().define_model_architecture()
        else:
            self.model = VanillaGenerator(self.params).to(self.device)
            self.params_trainable = list(filter(
                    lambda p: p.requires_grad, self.model.parameters()))
            n_trainable = sum(p.numel() for p in self.params_trainable)
            print(f"Number of generator parameters: {n_trainable}", flush=True)
        disc_type = self.params.get("disc_type", "FullyConnected")
        if disc_type == "ResNet":
            self.discriminator = ResNet(self.params).to(self.device)
        elif disc_type == "DenseNet":
            self.discriminator = DenseNet(self.params).to(self.device)
        elif disc_type == "FullyConnected":
            self.discriminator = AdversarialNet(self.params).to(self.device)
        else:
            raise(RuntimeError("Unknown Discriminator architecture: {}".format(disc_type)))
        disc_params = list(filter(
                lambda p: p.requires_grad, self.discriminator.parameters()))
        self.params_trainable = (self.params_trainable, disc_params)
        n_trainable = sum(p.numel() for p in disc_params)
        print(f"Number of discriminator parameters: {n_trainable}", flush=True)

        self.latent_loss    = LatentLoss(self.params)
        self.disc_loss      = GanLoss(self.params, self.data_store, adversarial=False)
        self.adv_loss       = GanLoss(self.params, self.data_store, adversarial=True)


    def get_optimizer(self, params, params_trainable):
        inn_optim = super().get_optimizer(params, params_trainable[0])
        optim_type = params.get("disc_optim_type", "Adam")
        lr = params.get("disc_lr", 0.0002)
        if optim_type == "Adam":
            disc_optim = torch.optim.AdamW(
                params_trainable[1],
                lr = lr,
                betas = params.get("disc_betas", [0.9, 0.999]),
                eps = params.get("disc_eps", 1e-8),
                weight_decay = params.get("disc_weight_decay", 1e-5)
            )
        elif optim_type == "SGD":
            disc_optim = torch.optim.SGD(
                params_trainable[1],
                lr = lr,
                momentum = params.get("disc_betas", [0.9, 0.999])[0],
                weight_decay = params.get("disc_weight_decay", 1e-5)
            )
        return inn_optim, disc_optim

    def get_scheduler(self, params, optim, train_loader):
        inn_lr_sched_mode, inn_scheduler = super().get_scheduler(params,
                                                        optim[0], train_loader)
        if inn_lr_sched_mode == "one_cycle_lr":
            #Adjust steps_per_epoch, as each epoch includes less updates
            inn_scheduler = OneCycleLR(
                optim[0],
                params["lr"],
                epochs = params["n_epochs"],
                steps_per_epoch=floor(len(train_loader)*params["batch_ratio"])+1)
        disc_lr_sched_mode = params.get("disc_lr_scheduler", "one_cycle_lr")
        if disc_lr_sched_mode == "step":
            disc_scheduler = torch.optim.lr_scheduler.StepLR(
                optim[1],
                step_size = params["disc_lr_decay_epochs"],
                gamma = params["disc_lr_decay_factor"],
            )
        elif disc_lr_sched_mode == "reduce_on_plateau":
            disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim[1],
                factor = 0.4,
                patience = 10,
                cooldown = 4,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif disc_lr_sched_mode == "one_cycle_lr":
            disc_scheduler = OneCycleLR(
                optim[1],
                params["disc_lr"],
                epochs = params["n_epochs"],
                steps_per_epoch=ceil(len(train_loader)*(1-params["batch_ratio"])))
        return inn_lr_sched_mode, inn_scheduler, disc_lr_sched_mode, disc_scheduler

    def set_optimizer(self):
        """Set the optimizer for training. ADAM is used with learning rate decay
        on plateau."""
        inn_optim, disc_optim = self.get_optimizer(self.params,
                                                             self.params_trainable)
        self.optim = (inn_optim, disc_optim)
        self.inn_lr_sched_mode, self.inn_scheduler, self.disc_lr_sched_mode, \
        self.disc_scheduler = self.get_scheduler(self.params,
                                                 self.optim,
                                                 self.train_loader)

    def add_provided_values(self, x, n_samples):
        for obs, value in self.provided_values:
            if obs not in x:
                x[obs] = torch.full((n_samples,), value, device=self.device,
                                    dtype=torch.float32)

    def change_format(self, x, forward=True):
        n_samples = x.shape[0]
        if forward:
            x = self.preproc.apply(x.squeeze(), forward=False, init_trafo=False)
            self.add_provided_values(x, n_samples)
            x = { obs: obs.from_data(x) for obs in self.obs_converter}
            x = self.disc_preproc.apply(x, forward=True, init_trafo=False, disc_steps=True)
        else:
            x = self.disc_preproc.apply(x, forward=False, init_trafo=False, disc_steps=True)
            self.add_provided_values(x, n_samples)
            x = { obs: obs.from_data(x) for obs in self.obs_converter_back}
            x = self.preproc.apply(x.squeeze(), forward=True, init_trafo=False)
        return x

    def disc_loss_old(self, noise_batch, x_samps, return_acc=False, requires_grad=True):
        noise_batch.requires_grad = requires_grad

        x_gen, jac = self.model(noise_batch, rev=True)
        x_gen = self.change_format(x_gen)
        x_samps = self.change_format(x_samps)

        x_samps.requires_grad = requires_grad

        pos = self.discriminator(x_samps)
        neg = self.discriminator(x_gen)

        regularization = 0
        if self.params.get("gradient_penalty", 0.0) > 0 and requires_grad:
            regularization = self.params.get("gradient_penalty", 0.0) * \
            Discriminator_Regularizer(pos, x_samps, neg, x_gen)

        if self.params.get("wasserstein", False):
            return torch.mean(pos - neg)
        else:
            if not return_acc:
                return self.BCE(neg, torch.zeros_like(neg)) + \
                self.BCE(pos, torch.ones_like(pos)) + regularization
            else:
                return self.BCE(neg, torch.zeros_like(neg)) + \
                self.BCE(pos, torch.ones_like(pos)) + regularization, \
                torch.mean((pos > 0).float()), torch.mean((neg < 0).float())

    def compute_pos_and_neg(self, noise_batch, x_samps):
        x_gen, jac = self.model(noise_batch, rev=True)
        x_gen = self.change_format(x_gen)
        x_samps = self.change_format(x_samps)
        pos = self.discriminator(x_samps)
        neg = self.discriminator(x_gen)
        return pos, neg, x_gen

    def validate(self, max_iterations=10):
        loss_gen, loss_disc, total_gauss_loss = 0, 0, 0
        max_val_batches = self.params["max_val_batches"]
        num_gen_iter = self.params.get("gen_batch_periter", 1)
        num_disc_iter = self.params.get("disc_batch_periter", 1)
        with torch.no_grad():
            test_iter = iter(self.test_loader)
            num_test_iters = floor(len(self.test_loader)\
            /(num_gen_iter + num_disc_iter))
            max_iter = min(num_test_iters, max_iterations)
            for epoch_iter in tqdm_verbose(range(max_iter), self.verbose,
                                    desc="Validation", leave=False, position=1):
                for gen_iter in range(num_gen_iter):
                    x_samps = test_iter.__next__()[0].to(self.device)
                    noise_batch = torch.randn(x_samps.shape).to(self.device)

                    z, jac = self.model(x_samps)
                    if self.epoch >= self.params.get("start_adv_training_epoch", 10):
                        sig = self.predict_discriminator(x_samps, sig = True).detach()
                        loss_gauss = self.latent_loss.apply(z, jac, sig)
                    else:
                        loss_gauss = self.latent_loss.apply(z, jac)
                    loss_gen += loss_gauss/(num_gen_iter*max_iter)
                    if self.params.get("lambda_adv", 1) > 0.0:
                        pos, neg, _ = self.compute_pos_and_neg(noise_batch, x_samps)
                        loss_gan = - self.adv_loss.apply(pos, neg, requires_grad=False,
                                                         epoch=self.epoch)

                        loss_gen += self.params.get("lambda_adv", 1) \
                        * loss_gan/(num_gen_iter*max_iter)

                    total_gauss_loss += loss_gauss/(num_gen_iter*max_iter)

                for disc_iter in range(num_disc_iter):
                    x_samps = test_iter.__next__()[0].to(self.device)
                    noise_batch = torch.randn(x_samps.shape).to(self.device)
                    pos, neg, _ = self.compute_pos_and_neg(noise_batch, x_samps)
                    loss = self.disc_loss.apply(pos, neg, requires_grad=False,
                                                epoch=self.epoch)
                    loss_disc += loss/(num_disc_iter*max_iter)
        return loss_gen, loss_disc, total_gauss_loss

    def write_epoch_summary(self, loss_mean, acc_mean, inn_lr, disc_lr):
        tqdm_write_verbose(f"Epoch: {self.epoch}", self.verbose)
        tqdm_write_verbose(f"Total Loss Generator:{loss_mean[2]:.6f}", self.verbose)
        tqdm_write_verbose(f"Total Loss Discriminator:{loss_mean[3]:.6f}", self.verbose)
        tqdm_write_verbose(f"Accuracy Truth Discriminator:{acc_mean[0]:.3f}", self.verbose)
        tqdm_write_verbose(f"Accuracy Fake Discriminator:{acc_mean[1]:.3f}", self.verbose)
        tqdm_write_verbose(f"Learning Rate Generator:{inn_lr:.2e}", self.verbose)
        tqdm_write_verbose(f"Learning Rate Discriminator:{disc_lr:.2e}", self.verbose)

    def train(self):
        """Train the model for n_epochs. During training the loss, learning rates
        and the model will be saved in intervals.
        """
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"]       = []
        self.data_store["disc_learning_rates"]  = []
        self.data_store["inn_losses"]           = []
        self.data_store["inn_loss_gan"]         = []
        self.data_store["inn_loss_gauss"]       = []
        self.data_store["disc_losses"]          = []
        self.data_store["weights_true"]         = []
        self.data_store["weights_fake"]         = []

        num_gen_iter    = self.params.get("gen_batch_periter", 1)
        num_disc_iter   = self.params.get("disc_batch_periter", 1)

        start_time      = time.time()
        self.eval_mode(evaluate=False)
        max_iter        = floor(len(self.train_loader)/(num_disc_iter + num_gen_iter))
        bestsofar       = 1e30


        for epoch in tqdm_verbose(range(self.params["n_epochs"]), self.verbose,
                                  desc="Epoch", leave=True, position=0):
            train_iter  = iter(self.train_loader)

            loss_mean   = [0, 0, 0, 0]
            acc_mean    = [0, 0]
            self.data_store["epoch_weights_true"] = []
            self.data_store["epoch_weights_fake"] = []

            for epoch_iter in tqdm_verbose(range(max_iter), self.verbose, desc="Epoch Iterations",
                        leave=False, position=1, total=len(self.train_loader)):
                for disc_iter in range(num_disc_iter):
                    self.optim[1].zero_grad()

                    x_samps = train_iter.__next__()[0].to(self.device)
                    x_samps.requires_grad = True
                    noise_batch = torch.randn(x_samps.shape).to(self.device)
                    noise_batch.requires_grad = True
                    pos, neg, x_gen = self.compute_pos_and_neg(noise_batch, x_samps)
                    loss_disc, acc_pos, acc_neg = self.disc_loss.apply(pos, neg,
                                                return_acc=True, epoch=self.epoch,
                                                x_samps=x_samps, x_gen=x_gen)

                    loss_disc.backward()
                    self.optim[1].step()

                    if self.disc_lr_sched_mode == "one_cycle_lr":
                        self.disc_scheduler.step()
                    acc_mean[0]  += acc_pos/((num_disc_iter)*max_iter)
                    acc_mean[1]  += acc_neg/((num_disc_iter)*max_iter)
                    loss_mean[3] += loss_disc.item()/(num_disc_iter*max_iter)

                    if not (loss_disc < 1e30):
                        print("Warning, DiscLoss of {} exceeds threshold, \
                        skipping back propagation".format(loss_disc.item()))
                        return

                for gen_iter in range(num_gen_iter):
                    x_samps = train_iter.__next__()[0].to(self.device)
                    noise_batch = torch.randn(x_samps.shape).to(self.device)
                    self.optim[0].zero_grad()

                    z, jac = self.model(x_samps)
                    if self.epoch >= self.params.get("start_adv_training_epoch", 10):
                        sig = self.predict_discriminator(x_samps, sig = True).detach()
                        loss_gauss = self.latent_loss.apply(z, jac, sig)
                    else:
                        loss_gauss = self.latent_loss.apply(z, jac)
                    loss_inn = 0 # Copy loss_gauss instead of reassigning the name
                    loss_inn += loss_gauss
                    if self.params.get("lambda_adv", 1) > 0.0:
                        pos, neg, _ = self.compute_pos_and_neg(noise_batch, x_samps)
                        loss_gan, acc_pos, acc_neg = self.adv_loss.apply(pos, neg,
                                                     return_acc=True, epoch=self.epoch)
                        # acc_mean[0]  += acc_pos/((num_gen_iter+num_disc_iter)*max_iter)
                        # acc_mean[1]  += acc_neg/((num_gen_iter+num_disc_iter)*max_iter)
                        loss_inn += self.params.get("lambda_adv", 1) \
                        * (-loss_gan)
                    else:
                        loss_gan = torch.Tensor([0])

                    if not (loss_inn < 1e30):
                        print("Warning, GenLoss of {} exceeds threshold, \
                        skipping back propagation".format(loss_inn.item()))
                        return

                    loss_inn.backward()
                    self.optim[0].step()
                    if self.inn_lr_sched_mode == "one_cycle_lr":
                        self.inn_scheduler.step()
                    loss_mean[0] += loss_gauss.item()/(num_gen_iter*max_iter)
                    loss_mean[1] -= loss_gan.item()/(num_gen_iter*max_iter)
                    loss_mean[2] += loss_inn.item()/(num_gen_iter*max_iter)

            if "weight_plots" in self.params.get("plots", []) and self.epoch%self.params.get("weight_interval",5) == 0:
                epoch_weights_true = np.array(self.data_store["epoch_weights_true"]).flatten()
                epoch_weights_fake = np.array(self.data_store["epoch_weights_fake"]).flatten()
                self.data_store["weights_true"].append(epoch_weights_true)
                self.data_store["weights_fake"].append(epoch_weights_fake)
            self.data_store["inn_losses"].append(loss_mean[2])
            self.data_store["inn_loss_gan"].append(loss_mean[1])
            self.data_store["inn_loss_gauss"].append(loss_mean[0])
            self.data_store["disc_losses"].append(loss_mean[3])

            self.epoch += 1

            #save the results of this epoch
            inn_lr = self.inn_scheduler.optimizer.param_groups[0]['lr']
            self.data_store["learning_rates"].append(inn_lr)
            disc_lr = self.disc_scheduler.optimizer.param_groups[0]['lr']
            self.data_store["disc_learning_rates"].append(disc_lr)

            self.write_epoch_summary(loss_mean, acc_mean, inn_lr, disc_lr)

            #handle learning rates
            gen_loss, disc_loss, total_gauss_loss = self.validate()
            if total_gauss_loss < bestsofar and self.params.get("save_best", False):
                tqdm_write_verbose(f"Saving new best model with gaussian loss \
                {total_gauss_loss:.6f} over {bestsofar:.6f}.", self.verbose)
                bestsofar = total_gauss_loss
                self.save(epoch="_best")

            #handle learning rates
            if self.inn_lr_sched_mode == "reduce_on_plateau":
                self.inn_scheduler.step(gen_loss)
            elif self.inn_lr_sched_mode == "step":
                self.inn_scheduler.step()
            if self.disc_lr_sched_mode == "reduce_on_plateau":
                self.disc_scheduler.step(gen_loss) #TODO: This should be disc_loss
            elif self.disc_lr_sched_mode == "step":
                self.disc_scheduler.step()


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

    def eval_mode(self, evaluate = True):
        """Set the model to eval mode if evaluate == True
        or to train mode if evaluate == False. This is needed so the whole
        dataset is used during testing. (As opposed to dropping the last batch)"""
        self.eval = evaluate
        if self.eval:
            return self.model.eval(), self.discriminator.eval()
        else:
            return self.model.train(), self.discriminator.train()

    def save(self, epoch=""):
        """Save the model, its optimizer and the test/train split, as well as the epoch"""
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({'opt':self.optim[0].state_dict(),
                    'net':self.model.state_dict(),
                    'epoch': self.epoch,
                    'disc': self.discriminator.state_dict(),
                    'disc_opt':self.optim[1].state_dict(),
                    'preproc':self.preproc.state_dict(),
                    'disc_preproc':self.disc_preproc.state_dict()},
                    self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch="", name = None):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        if name is None:
            name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts['net'])
        try:
            self.discriminator.load_state_dict(state_dicts['disc'])
        except:
            print("Warning: Could not load discriminator data from file {}".format(name))
        try:
            self.preproc.load_state_dict(state_dicts["preproc"])
        except Exception as e:
            print(e, "Warning: Could not load preproc datadata from file {}".format(name))
        try:
            self.disc_preproc.load_state_dict(state_dicts["disc_preproc"])
        except Exception as e:
            print(e, "Warning: Could not load discriminator preproc datadata from file {}".format(name))
        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print("Warning: Epoch number not provided in save file, setting to default {}".format(self.epoch))
        try:
            state_dicts["opt"]["param_groups"][0]["lr"] = self.params.get("lr", 0.0002)
            self.optim[0].load_state_dict(state_dicts['opt'])
        except Exception as e:
            print(e)
        try:
            state_dicts["disc_opt"]["param_groups"][0]["lr"] = self.params.get("disc_lr", 0.0002)
            self.optim[1].load_state_dict(state_dicts['disc_opt'])
        except Exception as e:
            print(e, "Warning: Could not load discriminator optimizer from file {}".format(name))
        self.model.to(self.device)
        self.discriminator.to(self.device)


    def predict_discriminator(self, data, sig, disc_format=False):
        if not disc_format:
            data = self.change_format(data)
        with torch.no_grad():
            try:
                labels = self.discriminator(data, sig=sig).squeeze()

            except RuntimeError:
                labels = []
                for i in range(ceil(len(data)/self.batch_size)):
                    batch = data[i*self.batch_size:(i+1)*self.batch_size]
                    labels.append(self.discriminator(batch, sig=sig).squeeze())

                labels = torch.cat(labels, dim=0)
            return labels

class VanillaGenerator(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.layers = []
        nlayers = params["layers_per_block"]
        dim_x = params["dim_x"]
        internal_size = params["internal_size"]

        for i in range(nlayers-1):
            self.layers.append(nn.Linear(dim_x if i == 0 else internal_size, internal_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=params.get("dropout", 0.0)))
        self.layers.append(nn.Linear(internal_size if nlayers > 1 else dim_x, dim_x))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, rev=True):
        if not rev:
            raise(ValueError("Can not run VanillaGenerator in phasespace -> gauss"))
        out = x
        for layer in self.layers:
            out = layer(out)
        return out



def weights_init(m, init_factor):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data *= init_factor

class AdversarialNet(nn.Module):

    def __init__(self, params):
        super().__init__()

        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act
        self.layers = []
        norm = lambda x: x
        nlayers = params["disc_layers"]
        dim_x = params["dim_disc"]
        internal_size = params["disc_internal_size"]

        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm

        for i in range(nlayers-1):
            self.layers.append(norm(nn.Linear(dim_x if i == 0 else internal_size, internal_size)))
            self.layers.append(activation())
            if params.get("batchnorm", False):
                self.layers.append(nn.BatchNorm1d(internal_size))
            self.layers.append(nn.Dropout(p=params.get("disc_dropout", 0.0)))
        self.layers.append(nn.Linear(internal_size if nlayers > 1 else dim_x, 1))
        self.layers = nn.Sequential(*self.layers)
        for layer in self.layers:
            weights_init(layer, params.get("disc_init_factor", 1))


    def forward(self, x, sig=False):
        x = self.layers(x)
        if sig:
            x = torch.sigmoid(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layers = []
        nlayers = params["disc_layers"]
        internal_size = params["disc_internal_size"]
        dim_x = params.get("intermediate_dim", 128)
        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act

        norm = lambda x: x
        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm

        for i in range(nlayers-1):
            self.layers.append(norm(nn.Linear(dim_x if i == 0 else internal_size, internal_size)))
            self.layers.append(activation())
            if params.get("batchnorm", False):
                self.layers.append(nn.BatchNorm1d(internal_size))
            self.layers.append(nn.Dropout(p=params.get("disc_dropout", 0.0)))
        self.layers.append(nn.Linear(internal_size if nlayers > 1 else dim_x, dim_x))
        self.layers = nn.ModuleList(self.layers)
        self.relu = activation()
        for layer in self.layers:
            weights_init(layer, params.get("disc_init_factor", 1))

    def forward(self, x):
        identity = x
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return self.relu(out + identity)


class ResNet(nn.Module):


    def __init__(self, params):
        super().__init__()
        self.blocks = []
        dim_x = params["dim_disc"]
        intermediate_dim = params.get("intermediate_dim", 128)
        n_blocks = params.get("disc_blocks", 8)
        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act
        norm = lambda x: x
        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm

        self.blocks.append(norm(nn.Linear(dim_x,intermediate_dim)))
        self.blocks.append(activation())
        for i in range(n_blocks-1):
            self.blocks.append(ResidualBlock(params))
        self.blocks.append(nn.Linear(intermediate_dim, 1))
        self.blocks = nn.ModuleList(self.blocks)
        for layer in self.blocks:
            weights_init(layer, params.get("disc_init_factor", 1))

    def forward(self, x, sig=False):
        for block in self.blocks:
            x = block(x)
        if sig:
            x = torch.sigmoid(x)
        return x

class LayerBlock(nn.Module):
    def __init__(self, params, dim_x, block_number):
        super().__init__()
        internal_size = params["disc_internal_size"]
        dim_x += block_number*internal_size
        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act

        norm = lambda x: x
        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm
        self.layers = []

        self.layers.append(norm(nn.Linear(dim_x, internal_size)))
        self.layers.append(activation())
        if params.get("batchnorm", False):
            self.layers.append(nn.BatchNorm1d(internal_size))
        self.layers.append(nn.Dropout(p=params.get("disc_dropout", 0.0)))
        self.layers = nn.ModuleList(self.layers)
        for layer in self.layers:
            weights_init(layer, params.get("disc_init_factor", 1))

    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(out)
        return out

class DenseBlock(nn.Module):
    def __init__(self, params, ):
        super().__init__()
        self.layers = []

        nlayers = params["disc_layers"]
        internal_size = params["disc_internal_size"]
        dim_x = params["dim_disc"] if not inner_block else params.get("intermediate_dim", 128)
        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act

        norm = lambda x: x
        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm

        for i in range(nlayers-1):
            self.layers.append(LayerBlock(params, dim_x, block_number=i))
        self.layers.append(norm(nn.Linear((internal_size)*(nlayers-1) + dim_x, dim_x)))
        self.layers = nn.ModuleList(self.layers)
        self.relu = activation()
        for layer in self.layers:
            weights_init(layer, params.get("disc_init_factor", 1))

    def forward(self, x):
        out_ = self.layers[0](x)
        out = x
        for layer in self.layers[1:]:
            out = torch.cat((out, out_), dim = -1)
            out_ = layer(out)
        return self.relu(out_)


class DenseNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.blocks = []
        norm = lambda x: x

        dim_x = params["dim_disc"]
        intermediate_dim = params.get("intermediate_dim", 128)
        n_blocks = params.get("disc_blocks", 8)
        act = getattr(nn, params.get("disc_activation", "ReLU"))
        if params.get("disc_activation", "ReLU") == "LeakyReLU":
            activation = lambda : act(params.get("negative_slope", 0.1))
        else:
            activation = act


        if params.get("disc_spectralnorm", False):
            norm = nn.utils.spectral_norm
        self.blocks.append(norm(nn.Linear(dim_x,intermediate_dim)))
        self.blocks.append(activation())
        for i in range(n_blocks-1):
            self.blocks.append(DenseBlock(params, inner_block=i>0))
        self.blocks.append(nn.Linear(intermediate_dim, 1))
        self.blocks = nn.ModuleList(self.blocks)
        for layer in self.blocks:
            weights_init(layer, params.get("disc_init_factor", 1))

    def forward(self, x, sig=False):
        for block in self.blocks:
            x = block(x)
        if sig:
            x = torch.sigmoid(x)
        return x
