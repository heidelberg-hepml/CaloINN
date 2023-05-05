import torch
import numpy as np
from math import floor, ceil
import time
import os

from multi_cond_inn import MultiCondINN
from igan import ResNet, DenseNet, AdversarialNet
from util import tqdm_verbose, tqdm_write_verbose, model_class, eval_observables_list, zero_nan_events
from losses import GanLoss, LatentLoss

@model_class
class MultiCondIGAN(MultiCondINN):
    def __init__(self, params, data_store, doc):
        super().__init__(params, data_store, doc)

        self.obs_converter = eval_observables_list(
            params.get("disc_observables", params["input_observables"]))
        self.obs_converter_back = eval_observables_list(params["input_observables"])
        self.provided_values = [(eval_observables_list(obs_expr), value)
                                for obs_expr, value in self.params.get("provided_values", [])]

    def define_model_architecture(self):
        super().define_model_architecture()

        default_params = self.params.get("discriminator_default", {})
        generator_dims = self.params["network_dims"]
        network_dims = self.params.get("disc_network_dims", generator_dims)
        print("Number of discriminator parameters:")
        for i, (ns, ndim, dim_gen) in enumerate(zip(self.network_sections,
                np.cumsum(network_dims), np.cumsum(generator_dims))):
            ns.disc_params = {**default_params, **self.params.get(f"discriminator_{i}", {})}
            print(ndim)
            ns.disc_params["dim_disc"] = ndim
            ns.dim_gen = dim_gen
            disc_type = ns.params.get("disc_type", "FullyConnected")
            if disc_type == "ResNet":
                ns.discriminator = ResNet(ns.disc_params).to(self.device)
            elif disc_type == "DenseNet":
                ns.discriminator = DenseNet(ns.disc_params).to(self.device)
            elif disc_type == "FullyConnected":
                ns.discriminator = AdversarialNet(ns.disc_params).to(self.device)
            else:
                raise(RuntimeError(f"Unknown Discriminator architecture: {disc_type}"))
            ns.disc_params_trainable = list(filter(
                    lambda p: p.requires_grad, ns.discriminator.parameters()))
            n_trainable = sum(p.numel() for p in ns.disc_params_trainable)
            print(f"    network {i}: {n_trainable}", flush=True)

            ns.disc_loss = GanLoss(ns.disc_params, self.data_store, adversarial=False)
            ns.latent_loss = LatentLoss(self.params)

    def set_optimizer(self):
        super().set_optimizer()
        for ns in self.network_sections:
            ns.disc_optimizer = self.get_optimizer(ns.disc_params, ns.disc_params_trainable)
            ns.disc_lr_sched_mode, ns.disc_scheduler = self.get_scheduler(
                    ns.disc_params, ns.disc_optimizer, ns.train_loader)

    def initialize_data_loaders(self):
        super().initialize_data_loaders()
        train_tensor = self.data_store["train_disc_preproc"].to(self.device)
        train_jet_count = train_tensor[:,0].long()
        jet_count_classes = self.max_jets - self.min_jets + 1

        for jet_count, net_section in zip(range(self.min_jets, self.max_jets+1),
                                          self.network_sections):
            train_mask = train_jet_count == jet_count
            train_dim = net_section.disc_params["dim_disc"]
            batch_size = net_section.params["batch_size"]
            net_section.disc_train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_tensor[train_mask, 1:1+train_dim]),
                batch_size = batch_size,
                drop_last = not self.eval,
                shuffle = True
            )

            net_section.jet_count = jet_count
            net_section.jet_sizes = torch.full((batch_size, 1), jet_count, device=self.device)
            if self.one_hot_count:
                net_section.jet_size_cond = torch.nn.functional.one_hot(
                        net_section.jet_sizes[:,0] - self.min_jets, jet_count_classes).float()
            else:
                net_section.jet_size_cond = net_section.jet_sizes.float()

    def train(self):
        """Train the model for n_epochs. During training the loss, learning rates
        and the model will be saved in intervals.
        """
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"]       = [[] for n in self.network_sections]
        self.data_store["disc_learning_rates"]  = [[] for n in self.network_sections]
        self.data_store["inn_losses"]           = [[] for n in self.network_sections]
        self.data_store["inn_loss_gan"]         = [[] for n in self.network_sections]
        self.data_store["inn_loss_gauss"]       = [[] for n in self.network_sections]
        self.data_store["disc_losses"]          = [[] for n in self.network_sections]
        self.data_store["weights_true"]         = [[] for n in self.network_sections]
        self.data_store["weights_fake"]         = [[] for n in self.network_sections]

        num_gen_iter    = self.params.get("gen_batch_periter", 1)
        num_disc_iter   = self.params.get("disc_batch_periter", 1)

        start_time  = time.time()
        sec_epochs  = []
        max_iter    = []

        bestsofar   = 1e30

        for net_sec in self.network_sections:
            net_sec.model.train()
            net_sec.discriminator.train()
            sec_epochs.append(net_sec.params["n_epochs"])
            if num_gen_iter == 0 and num_disc_iter != 0:
                max_iter.append(len(net_sec.disc_train_loader) // num_disc_iter)
            elif num_gen_iter != 0 and num_disc_iter == 0:
                max_iter.append(len(net_sec.train_loader) // num_gen_iter)
            elif (num_gen_iter, num_disc_iter) == (0,0):
                max_iter.append(0)
            else:
                max_iter.append(min(len(net_sec.train_loader) // num_gen_iter,
                                    len(net_sec.disc_train_loader) // num_disc_iter))

            
        for epoch in tqdm_verbose(range(max(sec_epochs)), self.verbose,
                                  desc="Epoch", leave=True, position=0):
            for i, (net_section, n_epo, ns_max_iter) in enumerate(zip(
                    self.network_sections, sec_epochs, max_iter)):
                gen_train_iter  = iter(net_section.train_loader)
                disc_train_iter = iter(net_section.disc_train_loader)

                #if self.epoch >= n_epo:
                #    continue

                loss_mean   = [0, 0, 0, 0]
                acc_mean    = [0, 0]
                w_mean      = [0, 0]
                self.data_store["epoch_weights_true"] = []
                self.data_store["epoch_weights_fake"] = []


                for epoch_iter in tqdm_verbose(range(ns_max_iter), self.verbose,
                        desc="Epoch Iterations", leave=False, position=1):
                    for disc_iter in range(num_disc_iter):
                        net_section.disc_optimizer.zero_grad()

                        x_samps = next(disc_train_iter)[0]
                        x_samps.requires_grad = True

                        noise_shape = (x_samps.shape[0], net_section.dim_gen)
                        noise_batch = torch.randn(noise_shape, device=self.device)
                        noise_batch.requires_grad = True
                        pos, neg, x_gen = self.compute_pos_and_neg(noise_batch, x_samps,
                                                                   net_section, i)
                        loss_disc, acc_pos, acc_neg = net_section.disc_loss.apply(pos, neg,
                                                return_acc=True, epoch=self.epoch,
                                                x_samps=x_samps, x_gen=x_gen)
                        if not (loss_disc < 1e30):
                            print("Warning, DiscLoss of {} exceeds threshold, \
                            skipping back propagation".format(loss_disc.item()))
                            return

                        loss_disc.backward()
                        net_section.disc_optimizer.step()



                        if net_section.disc_lr_sched_mode == "one_cycle_lr":
                            net_section.disc_scheduler.step()
                        acc_mean[0]  += acc_pos/(num_disc_iter*ns_max_iter)
                        acc_mean[1]  += acc_neg/(num_disc_iter*ns_max_iter)
                        loss_mean[3] += loss_disc.item()/(num_disc_iter*ns_max_iter)
                        w_mean[0] += torch.mean(torch.sigmoid(pos))/(num_disc_iter*ns_max_iter)
                        w_mean[1] += torch.mean(torch.sigmoid(neg))/(num_disc_iter*ns_max_iter)



                    for gen_iter in range(num_gen_iter):
                        x_samps = next(gen_train_iter)[0]
                        x_count, *x_rest = torch.split(x_samps, self.input_dims, dim=-1)
                        x_samps_split = (x_count[:, net_section.cond_mask], *x_rest)


                        net_section.optimizer.zero_grad()

                        z, jac = net_section.model(x_samps_split[i+1], c=x_samps_split[:i+1])

                        if self.epoch >= self.params.get("start_adv_training_epoch", 10):
                            net_section.latent_loss.weight_pot_scheduler(epoch)
                            sig = self.predict_discriminator(x_samps, sig = True, one_hot=True).detach()
                            loss_gauss = net_section.latent_loss.apply(z, jac, sig)
                        else:
                            loss_gauss = net_section.latent_loss.apply(z, jac)

                        loss_inn = 0 # Copy loss_gauss instead of reassigning the name
                        loss_inn += loss_gauss
                        lambda_adv = net_section.params.get("lambda_adv", 0.)
                        if lambda_adv > 0.0:
                            noise_shape = (x_count.shape[0], net_section.dim_gen)
                            noise_batch = torch.randn(noise_shape, device=self.device)
                            pos, neg, _ = self.compute_pos_and_neg(noise_batch,
                                                            x_samps, net_section)
                            loss_gan, acc_pos, acc_neg = self.adv_loss.apply(pos, neg,
                                                     return_acc=True, epoch=self.epoch)
                            # acc_mean[0]  += acc_pos/((num_gen_iter+num_disc_iter)*max_iter)
                            # acc_mean[1]  += acc_neg/((num_gen_iter+num_disc_iter)*max_iter)
                            loss_inn -= lambda_adv * loss_gan
                        else:
                            loss_gan = torch.Tensor([0])

                        if not (loss_inn < 1e30):
                            print("Warning, GenLoss of {} exceeds threshold, \
                            skipping back propagation".format(loss_inn.item()))
                            return

                        loss_inn.backward()
                        net_section.optimizer.step()


                        if net_section.lr_sched_mode == "one_cycle_lr":
                            net_section.scheduler.step()
                        loss_mean[0] += loss_gauss.item()/(num_gen_iter*ns_max_iter)
                        loss_mean[1] -= loss_gan.item()/(num_gen_iter*ns_max_iter)
                        loss_mean[2] += loss_inn.item()/(num_gen_iter*ns_max_iter)


                if "weight_plots" in self.params.get("plots", []) and self.epoch%self.params.get("weight_interval",5) == 0:
                    epoch_weights_true = np.array(self.data_store["epoch_weights_true"]).flatten()
                    epoch_weights_fake = np.array(self.data_store["epoch_weights_fake"]).flatten()
                    self.data_store["weights_true"][i].append(epoch_weights_true)
                    self.data_store["weights_fake"][i].append(epoch_weights_fake)
                self.data_store["inn_losses"][i].append(loss_mean[2])
                self.data_store["inn_loss_gan"][i].append(loss_mean[1])
                self.data_store["inn_loss_gauss"][i].append(loss_mean[0])
                self.data_store["disc_losses"][i].append(loss_mean[3])

                #save the results of this epoch
                inn_lr = net_section.scheduler.optimizer.param_groups[0]['lr']
                self.data_store["learning_rates"][i].append(inn_lr)
                disc_lr = net_section.disc_scheduler.optimizer.param_groups[0]['lr']
                self.data_store["disc_learning_rates"][i].append(disc_lr)

                tqdm_write_verbose(f"Epoch {self.epoch}, net {i}", self.verbose)
                tqdm_write_verbose(f"Total Loss Generator: {loss_mean[2]:.6f}", self.verbose)
                tqdm_write_verbose(f"Total Loss Discriminator: {loss_mean[3]:.6f}", self.verbose)
                tqdm_write_verbose(f"Accuracy Truth Discriminator: {acc_mean[0]:.3f}", self.verbose)
                tqdm_write_verbose(f"Accuracy Fake Discriminator: {acc_mean[1]:.3f}", self.verbose)
                tqdm_write_verbose(f"Weights Truth Discriminator: {w_mean[0]:.3f}", self.verbose)
                tqdm_write_verbose(f"Weights Fake Discriminator: {w_mean[1]:.3f}", self.verbose)
                tqdm_write_verbose(f"Learning Rate Generator: {inn_lr:.2e}", self.verbose)
                tqdm_write_verbose(f"Learning Rate Discriminator: {disc_lr:.2e}", self.verbose)

                #handle learning rates
                if net_section.lr_sched_mode == "reduce_on_plateau":
                    net_section.scheduler.step(loss_mean[2])
                elif net_section.lr_sched_mode == "step":
                    net_section.scheduler.step()
                if net_section.disc_lr_sched_mode == "reduce_on_plateau":
                    net_section.disc_scheduler.step(loss_mean[3])
                elif net_section.disc_lr_sched_mode == "step":
                    net_section.disc_scheduler.step()

            self.epoch += 1

            #TODO: do we need validation? if yes, uncomment this
            #handle learning rates
            #gen_loss, disc_loss, total_gauss_loss = self.validate()
            #if total_gauss_loss < bestsofar and self.params.get("save_best", False):
            #    tqdm_write_verbose(f"Saving new best model with gaussian loss \
            #    {total_gauss_loss:.6f} over {bestsofar:.6f}.", self.verbose)
            #    bestsofar = total_gauss_loss
            #    self.save(epoch="_best")

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
        final_loss = [self.data_store['inn_losses'][i][-1]
                      for i in range(len(self.network_sections))]
        print(f"Final train loss: {final_loss}")

    def add_provided_values(self, x, n_samples):
        for obs, value in self.provided_values:
            if obs not in x:
                x[obs] = torch.full((n_samples,), value, device=self.device,
                                    dtype=torch.float32)

    def change_format(self, x, forward=True, one_hot=False):
        n_samples = x.shape[0]
        if forward:
            if one_hot:
                m = self.max_jets - self.min_jets + 1
                jet_counts = (x[:,:m] == 1).nonzero(as_tuple=True)[1] + self.min_jets
                x = torch.cat((jet_counts[:,None], x[:,m:]), dim=1)
            x = self.preproc.apply(x.squeeze(), forward=False, init_trafo=False)
            self.add_provided_values(x, n_samples)
            x = { obs: obs.from_data(x) for obs in self.obs_converter}
            x = self.disc_preproc.apply(x, forward=True, init_trafo=False, disc_steps=True)
            n_removed, x = zero_nan_events(x)
            if n_removed > 0:
                print("Set {} entries with NaN value to zero, while changing the format.".format(n_removed), flush=True)
        else:
            x = self.disc_preproc.apply(x, forward=False, init_trafo=False, disc_steps=True)
            self.add_provided_values(x, n_samples)
            x = { obs: obs.from_data(x) for obs in self.obs_converter_back}
            x = self.preproc.apply(x.squeeze(), forward=True, init_trafo=False)
        return x

    def compute_pos_and_neg(self, noise_batch, x_samps, net_section, ns_index):
        with torch.no_grad():
            x_gen = []
            noise_split = torch.split(noise_batch, self.input_dims[1:ns_index+2], dim=-1)
            for k, ns in enumerate(self.network_sections[:ns_index+1]):
                cond_input = (net_section.jet_size_cond[:, ns.cond_mask], *x_gen)
                x_gen.append(ns.model(noise_split[k], c=cond_input, rev=True)[0])
            x_gen = torch.cat(x_gen, dim=1)

        x_gen.requires_grad = True    


        padding = torch.zeros((x_gen.shape[0], self.dim_x - x_gen.shape[1]),
                              device=self.device)
        x_gen_padded = torch.cat((net_section.jet_sizes, x_gen, padding), dim=1)
        x_gen_padded = self.change_format(x_gen_padded)
        x_gen = x_gen_padded[:,1:1+ns.disc_params["dim_disc"]]
        pos = net_section.discriminator(x_samps)
        neg = net_section.discriminator(x_gen)
        return pos, neg, x_gen

    def predict_discriminator(self, data, sig, disc_format=False, one_hot=False):
        if not disc_format:
            data = self.change_format(data, one_hot=one_hot)
        with torch.no_grad():
            labels = torch.zeros(len(data), device=data.device)
            for ns in self.network_sections:
                batch_size = ns.params["batch_size"]
                filtered_data = data[data[:,0] == ns.jet_count]
                indices = torch.where(data[:,0] == ns.jet_count)[0]
                for i in range(ceil(len(filtered_data) / batch_size)):
                    batch = filtered_data[i*batch_size:(i+1)*batch_size,
                                          1:1+ns.disc_params["dim_disc"]]
                    labels[indices[i*batch_size:(i+1)*batch_size]] = ns.discriminator(
                            batch, sig=sig).squeeze()
            return labels

    def save(self, epoch=""):
        """Save the model, its optimizer and the test/train split, as well as the epoch"""
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opts": [net_sec.optimizer.state_dict()
                             for net_sec in self.network_sections],
                    "nets": [net_sec.model.state_dict()
                             for net_sec in self.network_sections],
                    'epoch': self.epoch,
                    'discs': [net_sec.discriminator.state_dict()
                             for net_sec in self.network_sections],
                    'disc_opts':[net_sec.disc_optimizer.state_dict()
                                for net_sec in self.network_sections],
                    'preproc':self.preproc.state_dict(),
                    'disc_preproc':self.disc_preproc.state_dict()},
                    self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch="", name = None):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        if name is None:
            name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)
        for i, net_sec in enumerate(self.network_sections):
            net_sec.model.load_state_dict(state_dicts['nets'][i])
        try:
            for i, net_sec in enumerate(self.network_sections):
                net_sec.discriminator.load_state_dict(state_dicts['discs'][i])
        except:
            print("Warning: Could not load discriminator data from file {}".format(name))
        try:
            self.preproc.load_state_dict(state_dicts["preproc"])
        except Exception as e:
            print(e, "Warning: Could not load preproc datadata from file {}".format(name))
        #try:
        #    self.disc_preproc.load_state_dict(state_dicts["disc_preproc"])
        #except Exception as e:
        #   print(e, "Warning: Could not load discriminator preproc datadata from file {}".format(name))
        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print("Warning: Epoch number not provided in save file, setting to default {}".format(self.epoch))
        try:
            for i, net_sec in enumerate(self.network_sections):
                state_dicts["opts"][i]["param_groups"][0]["lr"] = self.params.get("lr", 0.0002)
                net_sec.optimizer.load_state_dict(state_dicts['opts'][i])
        except Exception as e:
            print(e)
        try:
            for i, net_sec in enumerate(self.network_sections):
                state_dicts["disc_opts"][i]["param_groups"][0]["lr"] = self.params.get("disc_lr", 0.0002)
                net_sec.disc_optimizer.load_state_dict(state_dicts['disc_opts'][i])
        except Exception as e:
            print(e, "Warning: Could not load discriminator optimizer from file {}".format(name))

        for net_sec in self.network_sections:
            net_sec.model.to(self.device)
            net_sec.discriminator.to(self.device)
