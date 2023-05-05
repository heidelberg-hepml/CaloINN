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
from hierarchical_coupling_block import HierarchicalCouplingBlock

@model_class
class MultiHierarchicalINN(INN):
    def get_coupling_block(self, params):
        return HierarchicalCouplingBlock, block_kwargs

    def define_model_architecture(self):
        self.network_dims = self.params["network_dims"]
        self.min_jets = self.params["min_jets"]
        self.max_jets = self.params["max_jets"]
        dims_in = sum(self.network_dims)
        dims_c = len(self.network_dims)
        self.input_dims = (dims_c, dims_in)

        if dims_in + 1 != self.dim_x:
            raise ValueError("Network dimensions don't sum up to total dimension")
        jet_count_classes = self.max_jets - self.min_jets + 1
        if dims_c != jet_count_classes:
            raise ValueError("Network dimensions inconsistent with min/max jet count")

        nodes = [ConditionNode(dims_c, name="cond"),
                 InputNode(dims_in, name="inp")]

        block_kwargs = {
            "input_split": self.network_dims,
            "subnet_constructor": self.get_constructor_func(self.params),
            "permute_soft" : self.params.get("permute_soft"),
            "num_bins": self.params.get("num_bins", 10),
            "bounds": self.params.get("bounds_init", 10),
            "combined_mode": self.params.get("combined_mode", False)
        }
        for i in range(self.params.get("n_blocks", 10)):
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    HierarchicalCouplingBlock,
                    block_kwargs,
                    conditions = nodes[0],
                    name = f"block_{i}"
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name='out'))
        self.model = ReversibleGraphNet(nodes, verbose=False).to(self.device)
        self.params_trainable = list(filter(
                lambda p: p.requires_grad, self.model.parameters()))
        n_trainable = sum(p.numel() for p in self.params_trainable)
        print(f"Number of generator parameters: {n_trainable}", flush=True)

    def initialize_data_loaders(self):
        """Set the model's train and test loader"""
        train_tensor = torch.Tensor(self.data_store["train_preproc"]).to(self.device)
        test_tensor = torch.Tensor(self.data_store["test_preproc"]).to(self.device)
        train_jet_count = train_tensor[:,0].long()
        test_jet_count = test_tensor[:,0].long()

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

        upsampling = self.params.get("upsampling", False)
        downsampling = self.params.get("downsampling", False)
        if upsampling or downsampling:
            train_kwargs = dict(sampler = BalancedSampler(
                train_jet_count - self.min_jets, jet_count_classes, upsampling))
            test_kwargs = dict(sampler = BalancedSampler(
                test_jet_count - self.min_jets, jet_count_classes, upsampling))
        else:
            train_kwargs = test_kwargs = dict(shuffle = True)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor),
            batch_size = self.params["batch_size"],
            drop_last = not self.eval,
            **train_kwargs
        )

        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_tensor),
            batch_size = self.params["batch_size"],
            drop_last = not self.eval,
            **test_kwargs
        )

    def train(self):
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
                x_count, x_data = torch.split(x_samps, self.input_dims, dim=-1)
                self.optim.zero_grad()
                z, jac = self.model(x_data, c=x_count)

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
        gauss_input = torch.randn((n_samples, self.dim_x-1)).to(self.device)
        jet_sizes = torch.from_numpy(np.random.choice(
                np.arange(self.min_jets, self.max_jets+1), (n_samples,1),
                replace = True, p = self.jet_priors)).to(self.device)
        jet_count_classes = self.max_jets - self.min_jets + 1
        jet_size_cond = torch.nn.functional.one_hot(jet_sizes[:,0] - self.min_jets,
                                                    jet_count_classes).float()

        split_input = torch.split(gauss_input, self.network_dims, dim=1)
        for i in range(jet_count_classes):
            mask = jet_sizes < self.min_jets + i
            split_input[i][mask.squeeze(),:] = 0.

        batch_size = self.params["batch_size"]
        events_predict = []
        with torch.no_grad():
            for i in tqdm_verbose(range(math.ceil(n_samples / batch_size)), self.verbose,
                                  desc="Generating events", leave=False):
                events_batch = self.model(gauss_input[i * batch_size:(i+1) * batch_size],
                                          c=jet_size_cond[i * batch_size:(i+1) * batch_size],
                                          rev=True)[0].squeeze()
                events_predict.append(events_batch)
            events_predict = torch.cat(events_predict, dim=0)
            events_predict = torch.cat((jet_sizes.float(), events_predict),
                                       dim=-1).cpu().detach().numpy()
        return events_predict

    def compute_latent(self):
        raise NotImplementedError()
