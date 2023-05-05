from FrEIA.framework import *
from FrEIA.modules import *
import numpy as np
import torch
import time
import math
import os
from types import SimpleNamespace

from multi_separate_inn import MultiSeparateINN
from util import tqdm_verbose, tqdm_write_verbose, BalancedSampler, model_class

@model_class
class MultiChainINN(MultiSeparateINN):
    def train(self):
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"] = [[] for n in self.network_sections]
        self.data_store["inn_losses"] = [[] for n in self.network_sections]

        start_time = time.time()
        for net_sec in self.network_sections:
            net_sec.model.train()

        for i, net_section in enumerate(self.network_sections):
            for epoch in tqdm_verbose(range(net_section.params["n_epochs"]), self.verbose,
                                      desc=f"Epoch (net {i})", leave=True, position=0):
                epoch_loss = 0
                for batch, (x_samps,) in tqdm_verbose(enumerate(net_section.train_loader),
                                                      self.verbose, desc="Batch",
                                                      leave=False, position=1,
                                                      total=len(net_section.train_loader)):
                    x_samps = x_samps.to(self.device)

                    net_section.optimizer.zero_grad()
                    z, jac = net_section.model(x_samps)
                    for pns in reversed(self.network_sections[:i]):
                        zn, jacn = pns.model(z[:,:pns.input_dim])
                        z = torch.cat((zn, z[:,pns.input_dim:]), dim=1)
                        jac += jacn
                    loss = torch.mean(z**2) / 2 - torch.mean(jac) / z.shape[1]
                    if not loss < 1e30:
                        print(f"Warning, Loss of {loss.item()} exceeds threshold, " +
                               "skipping back propagation")
                        return
                    epoch_loss += loss.item() / len(net_section.train_loader)

                    loss.backward()
                    net_section.optimizer.step()
                    if net_section.lr_sched_mode == "one_cycle_lr":
                        net_section.scheduler.step()

                if net_section.lr_sched_mode == "reduce_on_plateau":
                    net_section.scheduler.step(self.validate())
                elif net_section.lr_sched_mode == "step":
                    net_section.scheduler.step()

                self.data_store["inn_losses"][i].append(epoch_loss)
                epoch_lr = net_section.scheduler.optimizer.param_groups[0]['lr']
                self.data_store["learning_rates"][i].append(epoch_lr)
                tqdm_write_verbose(f"Model {i} Epoch {self.epoch}: " +
                        f"Loss={epoch_loss:.6f}, LR={epoch_lr:.2e}", self.verbose)

                self.epoch += 1

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

    def predict(self, n_samples):
        events_predict = []
        sample_sum = 0
        with torch.no_grad():
            for prior, input_dim, net_section, n_jets in zip(self.jet_priors, self.input_dims,
                    self.network_sections, range(self.min_jets, self.max_jets+1)):
                if n_jets == self.max_jets:
                    sec_samples = n_samples - sample_sum
                else:
                    sec_samples = round(n_samples * prior)
                sample_sum += sec_samples
                batch_size = net_section.params["batch_size"]
                gauss_input = torch.randn((sec_samples, input_dim)).to(self.device)
                for i in tqdm_verbose(range(math.ceil(sec_samples / batch_size)),
                                      self.verbose, desc=f"Generating events ({n_jets} jets)",
                                      leave=False):
                    z = gauss_input[i * batch_size:(i+1) * batch_size]
                    for pns in self.network_sections[:n_jets-self.min_jets]:
                        zn = pns.model(z[:,:pns.input_dim], rev=True)[0].squeeze()
                        z = torch.cat((zn, z[:,pns.input_dim:]), dim=1)
                    events_batch = net_section.model(z, rev=True)[0].squeeze()
                    events_padded = torch.zeros(events_batch.shape[0], self.dim_x,
                                                device=self.device)
                    events_padded[:, 0] = n_jets
                    events_padded[:, 1:events_batch.shape[1]+1] = events_batch
                    events_predict.append(events_padded)
            events_predict = torch.cat(events_predict, dim=0).cpu().detach().numpy()
        return events_predict
