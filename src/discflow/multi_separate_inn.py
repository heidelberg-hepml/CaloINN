from FrEIA.framework import *
from FrEIA.modules import *
import numpy as np
import torch
import time
import math
import os
from types import SimpleNamespace

from inn import INN
from util import tqdm_verbose, tqdm_write_verbose, BalancedSampler, model_class

@model_class
class MultiSeparateINN(INN):
    def define_model_architecture(self):
        self.input_dims = self.params["network_dims"]
        self.min_jets = self.params["min_jets"]
        self.max_jets = self.params["max_jets"]
        jet_count_classes = self.max_jets - self.min_jets + 1
        if len(self.input_dims) != jet_count_classes:
            raise ValueError("Network dimensions inconsistent with min/max jet count")

        default_params = self.params.get("network_params_default", {})
        self.network_sections = []
        print("Number of generator parameters:")
        for i, ndim in enumerate(self.input_dims):
            net_params = {**default_params, **self.params.get(f"network_params_{i}", {})}

            nodes = [InputNode(ndim, name=f"n{i}_input")]

            CouplingBlock, block_kwargs = self.get_coupling_block(net_params)
            for j in range(net_params.get("n_blocks")):
                nodes.append(Node([nodes[-1].out0], CouplingBlock, block_kwargs,
                                  name = f"n{i}_block{j}"))

            nodes.append(OutputNode([nodes[-1].out0], name = f"n{i}_out"))
            model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
            params_trainable = list(filter(
                    lambda p: p.requires_grad, model.parameters()))
            self.network_sections.append(SimpleNamespace(
                    params = net_params,
                    model = model,
                    params_trainable = params_trainable,
                    input_dim = ndim
            ))

            n_trainable = sum(p.numel() for p in params_trainable)
            print(f"    network {i}: {n_trainable}", flush=True)

    def set_optimizer(self):
        for net_section in self.network_sections:
            net_section.optimizer = self.get_optimizer(net_section.params,
                                                       net_section.params_trainable)
            net_section.lr_sched_mode, net_section.scheduler = self.get_scheduler(
                    net_section.params, net_section.optimizer, net_section.train_loader)

    def initialize_data_loaders(self):
        train_tensor = torch.Tensor(self.data_store["train_preproc"]).to(self.device)
        test_tensor = torch.Tensor(self.data_store["test_preproc"]).to(self.device)
        train_jet_count = train_tensor[:,0].long()
        test_jet_count = test_tensor[:,0].long()

        self.jet_priors = [torch.sum(train_jet_count==i).float().item()/len(train_tensor)
                           for i in range(self.min_jets, self.max_jets+1)]
        self.jet_priors[-1] = 1 - sum(self.jet_priors[:-1])

        for jet_count, input_dim, net_section in zip(range(self.min_jets, self.max_jets+1),
                                                     self.input_dims, self.network_sections):
            net_section.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    train_tensor[train_jet_count == jet_count, 1:input_dim+1]),
                batch_size = net_section.params["batch_size"],
                drop_last = not self.eval,
                shuffle = True
            )

            net_section.test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    test_tensor[test_jet_count == jet_count, 1:input_dim+1]),
                batch_size = net_section.params["batch_size"],
                drop_last = not self.eval,
                shuffle = True
            )

    def train(self):
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"] = [[] for n in self.network_sections]
        self.data_store["inn_losses"] = [[] for n in self.network_sections]

        start_time = time.time()
        sec_epochs = []
        for net_sec in self.network_sections:
            net_sec.model.train()
            sec_epochs.append(net_sec.params["n_epochs"])

        for epoch in tqdm_verbose(range(max(sec_epochs)), self.verbose,
                                  desc="Epoch", leave=True, position=0):

            for i, (net_section, n_epo) in enumerate(zip(self.network_sections,
                                                         sec_epochs)):
                if self.epoch >= n_epo:
                    continue

                epoch_loss = 0
                for batch, (x_samps,) in tqdm_verbose(enumerate(net_section.train_loader),
                                                      self.verbose, desc=f"Batch (net {i})",
                                                      leave=False, position=1,
                                                      total=len(net_section.train_loader)):
                    x_samps = x_samps.to(self.device)

                    net_section.optimizer.zero_grad()
                    z, jac = net_section.model(x_samps)
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
                    events_batch = net_section.model(
                        gauss_input[i * batch_size:(i+1) * batch_size], rev=True)[0].squeeze()
                    events_padded = torch.zeros(events_batch.shape[0], self.dim_x,
                                                device=self.device)
                    events_padded[:, 0] = n_jets
                    events_padded[:, 1:events_batch.shape[1]+1] = events_batch
                    events_predict.append(events_padded)
            events_predict = torch.cat(events_predict, dim=0).cpu().detach().numpy()
        return events_predict

    def compute_latent(self):
        raise NotImplementedError()

    def save(self, epoch=""):
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opts": [net_sec.optimizer.state_dict()
                             for net_sec in self.network_sections],
                    "nets": [net_sec.model.state_dict()
                             for net_sec in self.network_sections],
                    "epoch": self.epoch}, self.doc.get_file(f"model/model{epoch}", False))

    def load(self, epoch=""):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        name = self.doc.get_file(f"model/model{epoch}", False)
        state_dicts = torch.load(name, map_location=self.device)

        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print(f"Warning: Epoch number not provided in save file, setting to {self.epoch}")
        for net_sec, model_state, opt_state in zip(self.network_sections, state_dicts["nets"],
                                                   state_dicts["opts"]):
            net_sec.model.load_state_dict(model_state)
            net_sec.model.to(self.device)
            try:
                net_sec.optimizer.load_state_dict(opt_state)
            except ValueError as e:
                print(e)
