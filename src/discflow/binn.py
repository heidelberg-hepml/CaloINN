from vblinear import VBLinear
from inn import INN, SubnetConstructor
import torch
from util import tqdm_verbose, tqdm_write_verbose, model_class
import time
import math
import numpy as np

@model_class
class BINN(INN):
    def __init__(self, params, data_store, doc):
        super().__init__(params, data_store, doc)
        self.bayesian_layers = []

    def get_constructor_func(self, params):
        def func(x_in, x_out):
            subnet = SubnetConstructor(
                    params.get("layers_per_block", 3),
                    x_in, x_out,
                    internal_size = params.get("internal_size"),
                    dropout = params.get("dropout", 0.),
                    layer_class = VBLinear)
            self.bayesian_layers.extend(
                    layer for layer in subnet.layer_list if isinstance(layer, VBLinear))
            return subnet
        return func

    def train(self):
        """Train the model for n_epochs. During training the loss, learning rates
        and the model will be saved in intervals.
        """
        save_every = self.params.get("checkpoint_save_interval")
        save_overwrite = self.params.get("checkpoint_save_overwrite")

        self.data_store["learning_rates"] = []
        self.data_store["inn_losses"] = []
        self.data_store["kl_losses"] = []
        self.data_store["total_losses"] = []

        start_time = time.time()
        self.model.train()

        kl_scale = 1. / (len(self.train_loader) * self.params["batch_size"])
        loss_scale = 1. / len(self.train_loader)

        for epoch in tqdm_verbose(range(self.params["n_epochs"]), self.verbose,
                                  desc="Epoch", leave=True, position=0):
            epoch_inn_loss = 0.
            epoch_kl_loss = 0.
            epoch_total_loss = 0.

            for batch, (x_samps,) in tqdm_verbose(enumerate(self.train_loader), self.verbose,
                                                  desc="Batch", leave=False, position=1,
                                                  total=len(self.train_loader)):
                x_samps = x_samps.to(self.device)
                self.optim.zero_grad()
                z, jac = self.model(x_samps)

                inn_loss = torch.mean(z**2) / 2 - torch.mean(jac) / z.shape[1]
                kl_loss = kl_scale * sum(layer.KL() for layer in self.bayesian_layers)
                loss = inn_loss + kl_loss

                #if training diverges, stop
                if not loss < 1e30:
                    print(f"Warning, Loss of {loss.item()} exceeds threshold, " +
                           "skipping back propagation")
                    return

                #save losses
                epoch_inn_loss += inn_loss.item() * loss_scale
                epoch_kl_loss += kl_loss.item() * loss_scale
                epoch_total_loss += loss.item() * loss_scale

                loss.backward()
                self.optim.step()
                if self.lr_sched_mode == "one_cycle_lr":
                    self.scheduler.step()

            self.epoch += 1

            #save the results of this epoch
            self.data_store["inn_losses"].append(epoch_inn_loss)
            self.data_store["kl_losses"].append(epoch_kl_loss)
            self.data_store["total_losses"].append(epoch_total_loss)
            epoch_lr = self.scheduler.optimizer.param_groups[0]['lr']
            self.data_store["learning_rates"].append(epoch_lr)
            tqdm_write_verbose(f"Epoch {self.epoch}: total loss={epoch_total_loss:.4f}, " +
                               f"INN loss={epoch_inn_loss:.4f}, " +
                               f"KL loss={epoch_kl_loss:.4f}, " +
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
        print(f"Final train loss: {self.data_store['total_losses'][-1]:.4f}")

    def predict(self, n_samples):
        events_predict = []
        batch_size = self.params["batch_size"]
        weight_samples = self.params["weight_samples"]
        resample_latent = self.params.get("resample_latent", False)
        self.model.eval()
        with torch.no_grad():
            if not resample_latent:
                gauss_input = torch.randn((n_samples, self.dim_x)).to(self.device)
            for j in tqdm_verbose(range(weight_samples), self.verbose,
                                  desc="Generating events", leave=True, position=0):
                if resample_latent:
                    gauss_input = torch.randn((n_samples, self.dim_x)).to(self.device)
                [layer.reset_random() for layer in self.bayesian_layers]
                for i in tqdm_verbose(range(math.ceil(n_samples / batch_size)), self.verbose,
                                      desc=f"Weights {j}", leave=False, position=1):
                    events_batch = self.model(gauss_input[i * batch_size:(i+1) * batch_size],
                                              rev=True)[0].squeeze()
                    events_predict.append(events_batch)
            pred = torch.cat(events_predict, dim=0).cpu().detach().numpy()
            return pred
