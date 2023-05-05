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
class MultiCondINN(INN):
    def define_model_architecture(self):
        network_dims = self.params["network_dims"]
        self.min_jets = self.params["min_jets"]
        self.max_jets = self.params["max_jets"]
        self.one_hot_count = self.params.get("one_hot_count", True)
        if sum(network_dims) + 1 != self.dim_x:
            raise ValueError("Network dimensions don't sum up to total dimension")
        jet_count_classes = self.max_jets - self.min_jets + 1
        if len(network_dims) != jet_count_classes:
            raise ValueError("Network dimensions inconsistent with min/max jet count")
        self.input_dims = [jet_count_classes if self.one_hot_count else 1, *network_dims]

        default_params = self.params.get("network_params_default", {})
        self.network_sections = []
        print("Number of generator parameters:")
        for i, ndim in enumerate(network_dims):
            net_params = {**default_params, **self.params.get(f"network_params_{i}", {})}

            if self.one_hot_count:
                cond_mask = ([True] * (jet_count_classes-i) + [False] * i
                             if i != jet_count_classes-1 else [False] * jet_count_classes)
            else:
                cond_mask = [True] if i != jet_count_classes-1 else [False]

            nodes = [
                ConditionNode(sum(cond_mask), name=f"n{i}_cond0"),
                *[ConditionNode(dim, name=f"n{i}_cond{j+1}")
                  for j, dim in enumerate(self.input_dims[1:i+1])],
                InputNode(ndim, name=f"n{i}_input")
            ]

            CouplingBlock, block_kwargs = self.get_coupling_block(net_params)
            for j in range(net_params.get("n_blocks")):
                nodes.append(Node([nodes[-1].out0], CouplingBlock, block_kwargs,
                    conditions = nodes[:i+1], name = f"n{i}_block{j}"))

            nodes.append(OutputNode([nodes[-1].out0], name = f"n{i}_out"))
            model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
            params_trainable = list(filter(
                    lambda p: p.requires_grad, model.parameters()))
            self.network_sections.append(SimpleNamespace(
                    params = net_params,
                    model = model,
                    params_trainable = params_trainable,
                    cond_mask = cond_mask
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
        """Set the model's train and test loader"""
        train_tensor = self.data_store["train_preproc"].to(self.device)
        test_tensor = self.data_store["test_preproc"].to(self.device)
        train_jet_count = train_tensor[:,0].long()
        test_jet_count = test_tensor[:,0].long()

        if self.one_hot_count:
            jet_count_classes = self.max_jets - self.min_jets + 1
            train_tensor = torch.cat([
                torch.nn.functional.one_hot(train_jet_count - self.min_jets,
                                            jet_count_classes).float(),
                train_tensor[:,1:]], dim=1)
            test_tensor = torch.cat([
                torch.nn.functional.one_hot(test_jet_count - self.min_jets,
                                            jet_count_classes).float(),
                test_tensor[:,1:]], dim=1)

        self.jet_priors = [torch.sum(train_jet_count==i).float().item()/len(train_tensor)
                           for i in range(self.min_jets, self.max_jets+1)]
        self.jet_priors[-1] = 1 - sum(self.jet_priors[:-1])

        for jet_count, net_section in zip(range(self.min_jets, self.max_jets+1),
                                          self.network_sections):
            train_mask = train_jet_count >= jet_count
            test_mask = test_jet_count >= jet_count

            upsampling = net_section.params.get("upsampling", False)
            downsampling = net_section.params.get("downsampling", False)
            if upsampling or downsampling:
                num_counts = self.max_jets + 1 - jet_count
                train_kwargs = dict(sampler = BalancedSampler(
                    train_jet_count[train_mask] - jet_count, num_counts, upsampling))
                test_kwargs = dict(sampler = BalancedSampler(
                    test_jet_count[test_mask] - jet_count, num_counts, upsampling))
            else:
                train_kwargs = test_kwargs = dict(shuffle = True)

            net_section.train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_tensor[train_mask]),
                batch_size = net_section.params["batch_size"],
                drop_last = not self.eval,
                **train_kwargs
            )

            net_section.test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_tensor[test_mask]),
                batch_size = net_section.params["batch_size"],
                drop_last = not self.eval,
                **test_kwargs
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
                    x_count, *x_rest = torch.split(x_samps, self.input_dims, dim=-1)
                    x_samps_split = (x_count[:, net_section.cond_mask], *x_rest)

                    net_section.optimizer.zero_grad()
                    z, jac = net_section.model(x_samps_split[i+1], c=x_samps_split[:i+1])
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

        print("\nTraining complete")
        print(f"--- {time.time() - start_time:.0f} seconds ---")

    def predict(self, n_samples):
        gauss_input = torch.randn((n_samples, self.dim_x-1)).to(self.device)
        jet_sizes = torch.from_numpy(np.random.choice(
                np.arange(self.min_jets, self.max_jets+1), (n_samples,1),
                replace = True, p = self.jet_priors)).to(self.device)

        if self.one_hot_count:
            jet_count_classes = self.max_jets - self.min_jets + 1
            jet_size_cond = torch.nn.functional.one_hot(jet_sizes[:,0] - self.min_jets,
                                                        jet_count_classes).float()
        else:
            jet_size_cond = jet_sizes.float()

        gauss_input = torch.split(gauss_input, self.input_dims[1:], dim=-1)
        events_predict = [[] for n in self.network_sections]
        batch_size = self.params["network_params_default"]["batch_size"]
        with torch.no_grad():
            for k, net_section in enumerate(self.network_sections):
                for i in tqdm_verbose(range(math.ceil(n_samples / batch_size)),
                                      self.verbose, desc=f"Generating events (net {k})",
                                      leave=False):
                    cond_input = [jet_size_cond[i * batch_size:(i+1) * batch_size,
                                                net_section.cond_mask]]
                    cond_input += [events_predict[j][i] for j in range(k)]
                    events_batch = net_section.model(
                        gauss_input[k][i * batch_size:(i+1) * batch_size], cond_input,
                        rev=True
                    )[0].squeeze()
                    events_predict[k].append(events_batch)
            events_predict = [torch.cat(net_dim, dim = 0) for net_dim in events_predict]
            events_predict = torch.cat(events_predict, dim=-1)
            jet_dim = self.dim_x // self.max_jets
            for j in range(self.min_jets, self.max_jets):
                events_predict[(jet_sizes<j).squeeze(),j*jet_dim:] = 0
            events_predict = torch.cat((jet_sizes.float(), events_predict), dim=-1)
            assert events_predict.shape[-1] == self.dim_x
        return events_predict

    def compute_latent(self):
        raise NotImplementedError()
        #max_latent_batches = self.params.get("max_latent_batches", 10)
        #with torch.no_grad():
        #    [n.model.eval() for n in self.network_sections]
        #    rev_data = [[] for n in self.network_sections]
        #    for k, net_section in enumerate(self.network_sections):
        #        for batch, (x_samps,) in tqdm_verbose(enumerate(net_section.test_loader),
        #                                              self.verbose,
        #                                              desc="Calculating latent space",
        #                                              leave=False, position=1,
        #                                              total=max_latent_batches):
        #            if batch > max_latent_batches:
        #                break
        #            x_samps_split = torch.split(x_samps.to(self.device),
        #                                        self.input_dims, dim=-1)
        #            rev_data[k].append(net_section.model(x_samps_split[k+1],
        #                    c=x_samps_split[:k+1])[0].squeeze())
        #    rev_data = [torch.cat(net_dim, dim=0) for net_dim in rev_data]
        #    return torch.cat(rev_data, dim=-1).cpu().detach().numpy()

    def save(self, epoch=""):
        os.makedirs(self.doc.get_file("model", False), exist_ok=True)
        torch.save({"opts": [net_sec.optimizer.state_dict()
                             for net_sec in self.network_sections],
                    "nets": [net_sec.model.state_dict()
                             for net_sec in self.network_sections],
                    "epoch": self.epoch,
                    "preproc": self.preproc}, self.doc.get_file(f"model/model{epoch}", False))

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
        try:
            self.preproc.load_state_dict(state_dicts["preproc"])
        except Exception as e:
            print(e, "Warning: Could not load preproc datadata from file")
