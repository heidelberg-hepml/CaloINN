import sys

import os
import numpy as np

import torch

import data_util
from model import CINN, CVAE, KernelVAE
import plotting

from documenter import Documenter

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from myDataLoader import MyDataLoader
from copy import deepcopy
import atexit

from tqdm import tqdm


class VAETrainer:
    def __init__(self, params, device, doc, VAE_type="CVAE"):
        
        self.params = params
        self.device = device
        print("Device: ", self.device)
        self.doc = doc
        self.save_memory = params.get("save_memory", False)
        self.VAE_type = VAE_type
        

        # Load the data  
        self.train_loader, self.test_loader, self.layer_boundaries, self.negative_layers, self.coordinates = data_util.get_loaders(
            filename=params['data_path'],
            val_frac=params["val_frac"],
            batch_size=params.get('VAE_batch_size', None),
            used_layers=params.get("used_calo_layers", None),
            eps=params.get("eps", 1.e-10),
            device=device,
            drop_last=True,
            shuffle=True,
            save_memory=self.save_memory,
            width_noise=0)
        
        self.num_detector_layers = len(self.layer_boundaries) - 1
        self.data_dim = self.train_loader.data.shape[1]
        self.cond_dim = self.train_loader.cond.shape[1]
        
        data = self.train_loader.data
        cond = self.train_loader.cond
        
        # Create the VAE
        if VAE_type is not None:
            self.latent_dim = params["VAE_latent_dim"]
        else:
            self.latent_dim = data.shape[1] / 2
            
        if VAE_type == "CVAE":
            self.model = CVAE(input = data,
                            cond = cond,
                            latent_dim = self.latent_dim,
                            hidden_sizes = params["VAE_hidden_sizes"],
                            layer_boundaries_detector = self.layer_boundaries,
                            alpha = params.get("alpha", 1.e-6),
                            beta = params.get("VAE_beta", 1.e-5),
                            gamma = params.get("VAE_gamma", 1.e+3),
                            eps = params.get("eps", 1.e-10),
                            threshold=params.get("VAE_internal_threshold", False), 
                            sparsity_loss=params.get("sparsity_loss", None), 
                            learnable_norm=params.get("VAE_learnable_norm", False),
                            )
            
        elif VAE_type == "KVAE":
            self.model = KernelVAE(input = data,
                          cond = cond,
                          latent_dim = self.latent_dim,
                          hidden_sizes = params["VAE_hidden_sizes"],
                          hidden_sizes_kernel = params["VAE_hidden_sizes_kernel"],
                          kernel_size=params.get("VAE_kernel_size", 7),
                          kernel_stride=params.get("VAE_kernel_stride", 3),
                          kernel_latent=params.get("VAE_kernel_latent", 50),
                          layer_boundaries_detector = self.layer_boundaries,
                          alpha = params.get("alpha", 1.e-6),
                          beta = params.get("VAE_beta", 1.e-5),
                          gamma = params.get("VAE_gamma", 1.e+3),
                          eps = params.get("eps", 1.e-10),
                          threshold=params.get("VAE_internal_threshold", False), 
                          sparsity_loss=params.get("sparsity_loss", None), 
                          learnable_norm=params.get("VAE_learnable_norm", False),
                          )
        
        elif VAE_type is None:
            # We dont need the rest if no VAE is used
            return
                      
        else:
            raise NotImplementedError("Only CVAE and KVAE are implemented")

        self.model = self.model.to(self.device)
        
        self.logit_trafo_in = self.model.logit_trafo_in
        self.logit_trafo_out = self.model.logit_trafo_out
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Number of parameters", count_parameters(self.model))
                
        
        # Set the optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("VAE_lr", 1.e-4), weight_decay=self.params.get("VAE_weight_decay", 0.))
        
        # Configure a possible LR scheduler
        self.set_scheduler()
        
        # Print the model
        print(self.model)
        sys.stdout.flush()
        
        # Needed for documentation (printing & plotting)
        self.losses_train = {'bce': [], 'mae_logit': [], 'kl': [], 'sparsity': [], 'total': []}
        self.losses_test = {'bce': [], 'mae_logit': [], 'kl': [], 'sparsity': [], 'total': []}
        self.learning_rates = []
        self.max_grad = []
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0

    def set_scheduler(self):
        
        steps_per_epoch = len(self.train_loader)
        
        if self.params.get("VAE_lr_scheduler", None) == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size = self.params["VAE_lr_decay_epochs"],
                gamma = self.params["VAE_lr_decay_factor"],
            )
        elif self.params.get("VAE_lr_scheduler", None) == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                factor = 0.4,
                patience = 50,
                cooldown = 100,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif self.params.get("VAE_lr_scheduler", None) == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                self.params.get("VAE_max_lr", self.params["VAE_lr"]*10),
                epochs = self.params.get("VAE_opt_epochs") or self.params["VAE_n_epochs"],
                steps_per_epoch=steps_per_epoch)
            
        elif self.params.get("VAE_lr_scheduler", None) is None:
            self.scheduler = None

    def train(self):
        
        if self.VAE_type is None:
            print("No VAE is used. Only the INN will be trained.")
            return
        
        for epoch in tqdm(range(self.epoch_offset+1, self.params['VAE_n_epochs']+1)):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            
            # Do training and validation for the current epoch
            max_grad, train_loss, train_bce_loss, train_logit_loss, train_kl_loss, train_sparsity_loss = self.__train_one_epoch()
            test_loss, test_bce_loss, test_logit_loss, test_kl_loss, test_sparsity_loss = self.__do_validation()
            
                
            # Print the data saved for documentation
            self._print_losses(epoch, train_bce_loss, train_logit_loss, train_kl_loss, train_sparsity_loss, train_loss,
                              test_bce_loss, test_logit_loss, test_kl_loss, test_sparsity_loss, test_loss, max_grad)
            
            # Plot the losses as well
            if epoch >= 1:
                self._plot_losses()
                
            # If we reach the save interval, create all histograms for the observables,
            # plot the latent distribution and save the model
            if epoch % self.params.get("VAE_keep_models", self.params["VAE_n_epochs"]+1) == 0:
                self.save(epoch=epoch)
            
            if epoch % self.params.get("VAE_save_interval", 100) == 0 or epoch == self.params['VAE_n_epochs']:
                self.save()
                
                self.plot_results(epoch)         
         
    def __train_one_epoch(self):
        """Trains the model for one epoch. Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        train_loss = 0
        train_bce_loss = 0
        train_logit_loss = 0
        train_kl_loss = 0
        train_sparsity_loss = 0
        max_grad = 0.0

        # Set model to training mode
        self.model.train()
        
        # Iterate over all batches
        # x=data, c=condition
        for x, c in self.train_loader:
            
            if self.save_memory:
                x = x.to(self.device)
                c = c.to(self.device)
            
            # Initialize the gradient value for the documentation
            max_grad_batch = 0.0
            
            # Reset the optimizer
            self.optim.zero_grad()
            
            # Get the reconstruction loss
            loss, logit_loss, bce_loss, kl_loss, sparsity_loss  = self.model.reco_loss(x, c, zero_logit=self.params.get("VAE_zero_logit", True))
                
            # Calculate the gradients
            loss.backward()
            
            # Update the parameters
            self.optim.step()

            # Save the losses for documentation
            self.losses_train['bce'].append(bce_loss.item())
            self.losses_train['mae_logit'].append(logit_loss.item())
            self.losses_train['total'].append(loss.item())
            self.losses_train['kl'].append(kl_loss.item())
            self.losses_train['sparsity'].append(sparsity_loss.item())
            
            train_loss += loss.item()*len(x)
            train_bce_loss += bce_loss.item()*len(x)
            train_logit_loss += logit_loss.item()*len(x)
            train_kl_loss += kl_loss.item()*len(x)
            train_sparsity_loss += sparsity_loss.item()*len(x)
            
            # Save the LR if a scheduler is used
            if self.scheduler is not None:
                self.scheduler.step()
                self.learning_rates.append(self.scheduler.get_last_lr()[0])

            # Save the maximum gradient for documentation
            for param in self.model.parameters():
                if param.grad is not None:
                    max_grad_batch = max(max_grad_batch, torch.max(torch.abs(param.grad)).item())
            max_grad = max(max_grad_batch, max_grad)
            self.max_grad.append(max_grad_batch)
                
        # Normalize the losses to the dataset length and return them.
        # We need a different normalization here compared to the plotting because we summed over the whole epoch!
        train_bce_loss /= len(self.train_loader.data)
        train_logit_loss /= len(self.train_loader.data)
        train_kl_loss /= len(self.train_loader.data)
        train_sparsity_loss /= len(self.train_loader.data)
        train_loss /= len(self.train_loader.data)

        return max_grad, train_loss, train_bce_loss, train_logit_loss, train_kl_loss, train_sparsity_loss
                       
    def __do_validation(self):
        """Evaluates the model on the test set.
        Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        test_loss = 0
        test_bce_loss = 0
        test_logit_loss = 0
        test_kl_loss = 0
        test_sparsity_loss = 0
        
        # Evaluate the model on the test dataset and save the losses
        self.model.eval()
        with torch.no_grad():
            for x, c in self.test_loader:
                
                if self.save_memory:
                    x = x.to(self.device)
                    c = c.to(self.device)
                
                # Get the reconstruction loss
                loss, logit_loss, bce_loss, kl_loss, sparsity_loss  = self.model.reco_loss(x, c, zero_logit=self.params.get("VAE_zero_logit", True))
                
                # Save the losses
                test_loss += loss.item() * len(x)
                test_bce_loss += bce_loss.item() * len(x)
                test_logit_loss += logit_loss.item() * len(x)
                test_kl_loss += kl_loss.item() * len(x)
                test_sparsity_loss += sparsity_loss.item() * len(x)                
        
        # Normalize the losses for printing and plotting and store them also in the corresponding dict
        test_bce_loss /= len(self.test_loader.data)
        test_logit_loss /= len(self.test_loader.data)
        test_loss /= len(self.test_loader.data)
        test_kl_loss /= len(self.test_loader.data)
        test_sparsity_loss /= len(self.test_loader.data)
               
        self.losses_test['bce'].append(test_bce_loss)
        self.losses_test['mae_logit'].append(test_logit_loss)
        self.losses_test['total'].append(test_loss)
        self.losses_test['kl'].append(test_kl_loss)
        self.losses_test['sparsity'].append(test_sparsity_loss)

        return test_loss, test_bce_loss, test_logit_loss, test_kl_loss, test_sparsity_loss

    def get_reco(self, data, cond, batch_size=1000):
        
        if self.VAE_type is None:
            raise RuntimeError("Cannor reconstruct without a VAE model!")
        
        
        self.model.eval()
        reconstructed = torch.zeros((data.shape[0],data.shape[1]))
        
        with torch.no_grad():
            # Generate the data in batches according to batch_size
            for batch in range((data.shape[0]+batch_size-1)//batch_size):
                start = batch_size*batch
                stop = min(batch_size*(batch+1), data.shape[0])
                cond_l = cond[start:stop].to(self.device)
                data_l = data[start:stop].to(self.device)
                reconstructed[start:stop] = self.model(data_l, cond_l).cpu()
            
        reconstructed = reconstructed[:,...]
        
        return reconstructed

    def encode(self, data, cond, batch_size=1000):
        
        if self.VAE_type is not None:
            self.model.eval()
        
        num_samples = len(data)
        
        with torch.no_grad():
                       
            # Prepares an "empty" container for the samples
            samples = torch.zeros((num_samples,int(self.latent_dim*2)+self.num_detector_layers))
            
            for batch in range((num_samples+batch_size-1)//batch_size):
                start = batch_size*batch
                stop = min(batch_size*(batch+1), num_samples)
                
                data_l = data[start:stop].to(self.device)
                cond_l = cond[start:stop].to(self.device)
                
                if self.VAE_type is not None:
                    
                    # VAE encoding
                    mu_l, logvar_l = self.model.encode(data_l, cond_l)
                    vae_latent = torch.cat((mu_l, logvar_l), axis=1)
                    extra_dims = self.logit_trafo_in(cond_l[:, 1:-self.num_detector_layers])*0.9  # TODO: Might use clipping here, instead of the 0.9...
                    
                
                else:
                    vae_latent = data_util.normalize_layers(data_l, self.layer_boundaries, c=cond_l, eps=self.params.get("eps", 1.e-10))
                    extra_dims = cond_l[:, 1:-self.num_detector_layers] 
                
                             
                # Append the extra dims that are learned by the INN to the training data
                samples_l = torch.cat((vae_latent, extra_dims), axis=1)
                
                # Fill the results array
                samples[start:stop] = samples_l.cpu()
                                
            return samples, cond[..., [0]].cpu()
   
    def encode_loaders(self, delete_old_loaders=True):
        
        # Encode the data
        data_train, cond_train = self.encode(self.train_loader.data, self.train_loader.cond)
        data_test, cond_test = self.encode(self.test_loader.data, self.test_loader.cond)
        
        if delete_old_loaders:
            # Delete the old loaders to make space for the new ones
            del self.train_loader, self.test_loader
    
        # Create the dataloaders
        batch_size = self.params.get('batch_size')
        loader_train = MyDataLoader(data_train.to(self.device), cond_train.to(self.device), batch_size, width_noise=self.params.get("width_noise", 0.0))
        loader_test = MyDataLoader(data_test.to(self.device), cond_test.to(self.device), batch_size, width_noise=self.params.get("width_noise", 0.0))
    
        if self.params.get("width_noise", 0.0) > 0:
            print("Adding noise to the INN loaders")
        
        return loader_train, loader_test, self.layer_boundaries, self.negative_layers, self.coordinates
     
    def _get_full_cond(self, e_inc, extra_dims):
        """Recreate the full VAE cond data from the extra dims and the incident energy. Therefore the 
        layer energies will be calculated from the extra dims and the incident energy."""
        layer_energies = []
        
        e_tot = extra_dims[..., [0]] * e_inc
        
        extra_dims_list = []
        
        for layer in range(self.num_detector_layers-1):
            
            if layer == 0:
                layer_energy = (e_tot) * extra_dims[..., [layer+1]]
                cumsum_previous_layers = torch.clone(layer_energy)
            else:
                layer_energy = (e_tot - cumsum_previous_layers) * extra_dims[..., [layer+1]] 
                cumsum_previous_layers += layer_energy
                
            layer_energies.append(layer_energy)
            extra_dims_list.append(extra_dims[..., [layer]])
            
        layer_energies.append(e_tot - cumsum_previous_layers)
        extra_dims_list.append(extra_dims[..., [layer+1]])
        
        layer_energies = torch.cat([e_inc] + extra_dims_list + layer_energies, axis=1)
        
        return layer_energies  

    def decode(self, latent, e_inc, batch_size=1000):
        
        if self.VAE_type is not None:
            self.model.eval()
        
        with torch.no_grad():

            # For the INN the energy dimensions are part of the training set.
            # For the VAE they are part of the conditioning. So we have to slice them off and
            # append them to the existing E_inc condition.

            latent_dim = self.latent_dim
            num_samples = latent.shape[0]
            
            # Prepares an "empty" container for the samples
            samples = torch.zeros((num_samples,self.data_dim))
            condition = torch.zeros((num_samples, self.cond_dim))
            
            
            
            for batch in range((num_samples+batch_size-1)//batch_size):
                start = batch_size*batch
                stop = min(batch_size*(batch+1), num_samples)
                
                if self.VAE_type is not None:
                    # VAE version used a logit for the cond preprocessing
                    extra_dims_l = self.logit_trafo_out(latent[start:stop, -self.num_detector_layers:].to(self.device))/0.9  # TODO: Might use clipping here, instead of the 0.9...
                else:
                    extra_dims_l = latent[start:stop, -self.num_detector_layers:].to(self.device)
                    
                    
                e_inc_l = e_inc[start:stop].to(self.device)
                condition_l = self._get_full_cond(e_inc=e_inc_l, extra_dims=extra_dims_l)
                
                if self.VAE_type is not None:
                    # VAE decoding                    
                    mu_l = latent[start:stop, :latent_dim].to(self.device)
                    logvar_l = latent[start:stop, latent_dim:-self.num_detector_layers].to(self.device)
                    
                    reparametrized_samples_latent_l = self.model.reparameterize(mu_l, logvar_l)
                    samples_l = self.model.decode(latent=reparametrized_samples_latent_l, c=condition_l)
                    
                else:
                    latent_l = latent[start:stop, :int(2*latent_dim)].to(self.device)
                    latent_l[latent_l < self.params.get("width_noise", 0.0)] = 0
                    samples_l = data_util.unnormalize_layers(latent_l, condition_l, self.layer_boundaries, eps=self.params.get("eps", 1.e-10), noise_width=None)
                
                
                # Fill the results array
                samples[start:stop] = samples_l.cpu()
                condition[start:stop] = condition_l.cpu()
                
            return samples, condition
           
    def plot_results(self, epoch, plot_path=None):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
                
        if self.VAE_type is None:
            print("Nothing to plot")
            return
        
        self.model.eval()

        # Generate the reconstructions
        data = self.test_loader.data
        cond = self.test_loader.cond
        generated = self.get_reco(data, cond)
        
        # make plots for the all energies and for the seperate incident energies, as well
        energies = torch.unique(cond[:, 0]).to(cond.device)
        masks = [torch.ones_like(cond[:, 0], dtype=torch.bool, device=cond.device)]
        for energy in energies:
            masks.append(cond[:, 0] == energy)
            
        for i, mask in enumerate(masks):
            
            # Postprocess the data
            data_post, cond_post, layer_boundaries_post = data_util.postprocess(data[mask], cond[mask], self.layer_boundaries, self.negative_layers)
            generated_post, _, _ = data_util.postprocess(generated[mask], cond[mask], self.layer_boundaries, self.negative_layers)
            
            # Get the plot paramters
            params = plotting.get_plot_params(layer_boundaries_post, self.coordinates.cpu().numpy(), used_layers=self.params.get("used_calo_layers", None), short=(i!=0))
                
                        
            # Now create the histograms
            if plot_path is None:
                subdir = os.path.join("plots", f'epoch_{epoch:03d}')
                plot_dir = self.doc.get_file(subdir)
            else:
                plot_dir = plot_path
            
            if i == 0:
                name = "summary_all_eincs.pdf"
            else:
                name = f"summary_{energies[i-1].item()}_MeV.pdf"
                
            plotting.plot_all_hist([data_post.cpu().numpy(), generated_post.cpu().numpy()], 
                                [cond_post.cpu().numpy(), cond_post.cpu().numpy()], 
                                params, plot_dir=plot_dir, summary_plot=True,
                                summary_plot_name=name, errorbars_true=True,
                                errorbars_fake=True, ncol=5)
            
    def _plot_losses(self):
        # Plot the losses
        plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
        plotting.plot_loss(self.doc.get_file('loss_bce.pdf'), self.losses_train['bce'], self.losses_test['bce'])
        plotting.plot_loss(self.doc.get_file('loss_logit.pdf'), self.losses_train['mae_logit'], self.losses_test['mae_logit'])
        plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
        plotting.plot_loss(self.doc.get_file('loss_sparsity.pdf'), self.losses_train['sparsity'], self.losses_test['sparsity'])
        
        # Plot the learning rate (if we use a scheduler)
        if self.scheduler is not None:
            plotting.plot_lr(self.doc.get_file('learning_rate.pdf'), self.learning_rates, len(self.train_loader))
        
        # Plot the gradients
        plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))

    def _print_losses(self, epoch, train_bce_loss, train_logit_loss, train_kl_loss, train_sparsity_loss, train_loss, 
                     test_bce_loss, test_logit_loss, test_kl_loss, test_sparsity_loss, test_loss, max_grad):
        print('')
        print(f'=== epoch {epoch} ===')
        
        print(f'bce loss (train): {train_bce_loss}')
        print(f'logit-loss (train): {train_logit_loss}')
        print(f'kl loss (train): {train_kl_loss}')
        print(f'sparsity loss (train): {train_sparsity_loss}')
        print(f'total loss (train): {train_loss}')
        
        print(f'bce loss (test): {test_bce_loss}')
        print(f'logit-loss (test): {test_logit_loss}')
        print(f'kl loss (test): {test_kl_loss}')
        print(f'sparsity loss (test): {test_sparsity_loss}')
        print(f'total loss (test): {test_loss}')
        
        if self.scheduler is not None:
                print(f'lr: {self.scheduler.get_last_lr()[0]}')

        print(f'maximum gradient: {max_grad}')
        
        sys.stdout.flush()
         
    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses and the epoch """
        
        if self.VAE_type is None:
            return
        
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses_test": self.losses_test,
                    "losses_train": self.losses_train,
                    "grads": self.max_grad,
                    "epoch": self.epoch,
                    "learning_rates": self.learning_rates,}, 
                   
                   self.doc.get_file(f"model{epoch}.pt"))
                         
    def load(self, epoch="", update_offset=True):
        """ Load the model, its optimizer, losses and the epoch """
        
        if self.VAE_type is None:
            return
        
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        self.losses_test = state_dicts.get("losses_test", {})
        self.losses_train = state_dicts.get("losses_train", {})
        self.epoch = state_dicts.get("epoch", 0)
        self.max_grad = state_dicts.get("grads", [])
        self.learning_rates = state_dicts.get("learing_rates", [])
        if update_offset:
            self.epoch_offset = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)
        print(f"loaded VAE state from epoch {state_dicts.get('epoch', 0)}")
                                                                                                                     
       
class ECAETrainer:
    def __init__(self, params, device, doc, vae_dir=None):
        
        # Save some important paramters
        self.params = params
        self.device = device
        self.doc = doc
        
        self.VAE_type = params.get("VAE_type", "CVAE")
        if self.VAE_type is None:
            vae_basedir = os.path.join(doc.basedir, "VAE")
            vae_doc = Documenter(params['run_name'], existing_run=True, basedir=vae_basedir, log_name="log_jupyter.txt", read_only=True)
            self.preprocessor = VAETrainer(params, device, vae_doc, VAE_type=self.VAE_type)
            
        else:
            # Create a VAE trainer, train it and make sure, that the plots of the VAE are put in a different directory
            if vae_dir is None:
                vae_basedir = os.path.join(doc.basedir, "VAE")
                vae_doc = Documenter(params['run_name'], existing_run=True, basedir=vae_basedir, log_name="log_jupyter.txt", read_only=True)
                self.preprocessor = VAETrainer(params, device, vae_doc, VAE_type=self.VAE_type)
                print("\n\nStart training of CVAE\n\n")
                self.preprocessor.train()
                print("\n\nEnd training of CVAE\n\n")
            
            else:
                vae_doc = Documenter(params['run_name'], existing_run=True, basedir=vae_dir, log_name="log_jupyter.txt", read_only=True)
                self.preprocessor = VAETrainer(params, device, vae_doc, VAE_type=self.VAE_type)

            self.preprocessor.load()
                        
        self.train_loader, self.test_loader, self.layer_boundaries, self.negative_layers, self.coordinates = self.preprocessor.encode_loaders()
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0
        
        # Save the dataloaders ("test" should rather be called validation...)
        
        # Whether the last batch should be dropped if it is smaller
        if self.params.get("drop_last", False):
            self.train_loader.drop_last_batch()
            self.test_loader.drop_last_batch()
            
        # Save the input dimention of the model == latentspace of the VAE + 3
        self.num_dim = self.train_loader.data.shape[1]
        print(f"Input dimension: {self.num_dim}")

        # Initialize the model with the full data to get the dimensions right
        data = torch.clone(self.train_loader.data)
        cond = torch.clone(self.train_loader.cond)
        model = CINN(params, data, cond)
        self.model = model.to(device)
        print(self.model)
        
        # Initialize the optimizer and the learning rate scheduler
        # Default: Adam & reduce on plateau
        self.set_optimizer(steps_per_epoch=len(self.train_loader))

        # Create some empty containers for the losses and gradients.
        # Needed for documentation (printing & plotting)
        self.losses_train = {'inn': [], 'kl': [], 'total': []}
        self.losses_test = {'inn': [], 'kl': [], 'total': []}
        self.learning_rates = []
        self.max_grad = []
        self.grad_norm = []
        
        # only needed for a bayesian setup, but makes save and load easier if always created
        self.min_logsig = []
        self.max_logsig = []
        self.mean_logsig = []
        self.median_logsig = []
        self.close_to_prior = []
        

        if self.model.bayesian:
            # save the prior as logsig2 value for later usage
            self.logsig2_prior = - np.log(params.get("prior_prec", 1))
                          
    def train(self):
        """ Trains the model. """

        # Deactivated with reset random -> Not active for plot uncertainties
        if self.model.bayesian:
            self.model.enable_map()

        # Plot some images of the latent space
        # Want to check that it converges to a gaussian.
        self.latent_samples(0)

        print("\n\nstart the training of the INN\n\n")
        
        # Start the actual training
        for epoch in tqdm(range(self.epoch_offset+1,self.params['n_epochs']+1)):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            
            # Do training and validation for the current epoch
            if self.model.bayesian:
                max_grad, train_loss, train_inn_loss, train_kl_loss = self.__train_one_epoch()
                test_loss, test_inn_loss, test_kl_loss = self.__do_validation()
                max_bias, max_mu_w, min_logsig2_w, max_logsig2_w = self.__analyze_logsigs()
            else:       
                max_grad, train_loss, train_inn_loss = self.__train_one_epoch()
                test_loss, test_inn_loss = self.__do_validation()
                                
            # Print the data saved for documentation
            print('')
            print(f'=== epoch {epoch} ===')
            print(f'inn loss (train): {train_inn_loss}')
            if self.model.bayesian:
                print(f'kl loss (train): {train_kl_loss}')
                print(f'total loss (train): {train_loss}')
            print(f'inn loss (test): {test_inn_loss}')
            if self.model.bayesian:
                print(f'kl loss (test): {test_kl_loss}')
                print(f'total loss (test): {test_loss}')
            if self.scheduler is not None:
                print(f'lr: {self.scheduler.get_last_lr()[0]}')
            if self.model.bayesian:
                print(f'maximum bias: {max_bias}')
                print(f'maximum mu_w: {max_mu_w}')
                print(f'minimum logsig2_w: {min_logsig2_w}')
                print(f'maximum logsig2_w: {max_logsig2_w}')
            print(f'maximum gradient: {max_grad}')
            sys.stdout.flush()
            
            
            # Plot the data saved for documentation
            if epoch >= 1:
                # Plot the losses
                plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
                if self.model.bayesian:
                    plotting.plot_loss(self.doc.get_file('loss_inn.pdf'), self.losses_train['inn'], self.losses_test['inn'])
                    plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
                    plotting.plot_logsig(self.doc.get_file('logsig_2.pdf'),
                                         [self.max_logsig, self.min_logsig, self.mean_logsig, self.median_logsig])
                    
                # Plot the learning rate (if we use a scheduler)
                if self.scheduler is not None:
                    plotting.plot_lr(self.doc.get_file('learning_rate.pdf'), self.learning_rates, len(self.train_loader))
                
                # Plot the gradients
                plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))
                if self.params.get("store_grad_norm", True):
                    plotting.plot_grad(self.doc.get_file('gradient_norm.pdf'), self.grad_norm, len(self.train_loader))

            # If we reach the save interval, create all histograms for the observables,
            # plot the latent distribution and save the model
            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                if epoch % self.params.get("keep_models", self.params["n_epochs"]+1) == 0:
                    self.save(epoch=epoch)
                self.save()
                
                self.latent_samples(epoch)
                self.plot_results(epoch)
     
    def __train_one_epoch(self):
        """Trains the model for one epoch. Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        train_loss = 0
        train_inn_loss = 0
        if self.model.bayesian:
            train_kl_loss = 0
        max_grad = 0.0

        # Set model to training mode
        self.model.train()
        
        # Iterate over all batches
        # x=data, c=condition
        for x, c in self.train_loader:
            
            # Initialize the gradient value for the documentation
            max_grad_batch = 0.0
            
            # Reset the optimizer
            self.optim.zero_grad()
            
            # Get the log likelihood loss
            inn_loss = - torch.mean(self.model.log_prob(x,c))
            
            # For a bayesian setup add the properly normalized kl loss term
            # Otherwise only use the inn_loss
            if self.model.bayesian:
                kl_loss = self.model.get_kl() / len(self.train_loader.data) # must normalize for consistency
                loss = inn_loss + kl_loss
                
                # Save the loss for documentation
                self.losses_train['kl'].append(kl_loss.item())
                train_kl_loss += kl_loss.item()*len(x)
            else:
                loss = inn_loss
                
            # Calculate the gradients
            loss.backward()
                
            # Update the parameters
            self.optim.step()

            # Save the losses for documentation
            self.losses_train['inn'].append(inn_loss.item())
            self.losses_train['total'].append(loss.item())
            train_inn_loss += inn_loss.item()*len(x)
            train_loss += loss.item()*len(x)
            
            # Save the LR if a scheduler is used
            if self.scheduler is not None:
                self.scheduler.step()
                self.learning_rates.append(self.scheduler.get_last_lr()[0])

            # Save the maximum gradient for documentation
            for param in self.model.params_trainable:
                if param.grad is not None:
                    max_grad_batch = max(max_grad_batch, torch.max(torch.abs(param.grad)).item())
            max_grad = max(max_grad_batch, max_grad)
            self.max_grad.append(max_grad_batch)
            
            # Save the gradient value corresponding to the L2 norm
            if self.params.get("store_grad_norm", True):
                grads = [p.grad for p in self.model.params_trainable if p.grad is not None] 
                norm_type = float(2) # L2 norm
                total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(self.device) for g in grads]), norm_type)
                self.grad_norm.append(total_norm.item())
                
        # Normalize the losses to the dataset length and return them.
        # We need a different normalization here compared to the plotting because we summed over the whole epoch!
        train_inn_loss /= len(self.train_loader.data)
        train_loss /= len(self.train_loader.data)                
            
        if self.model.bayesian:
            train_kl_loss /= len(self.train_loader.data)
            return max_grad, train_loss, train_inn_loss, train_kl_loss
                
        return max_grad, train_loss, train_inn_loss
                
    def __do_validation(self):
        """Evaluates the model on the test set.
        Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        test_loss = 0
        test_inn_loss = 0
        if self.model.bayesian:
            test_kl_loss = 0
        
        # Evaluate the model on the test dataset and save the losses
        self.model.eval()
        with torch.no_grad():
            for x, c in self.test_loader:
                
                inn_loss = - torch.mean(self.model.log_prob(x,c))
                if self.model.bayesian:
                    kl_loss = self.model.get_kl() / len(self.train_loader.data) # must normalize for consistency
                    loss = inn_loss + kl_loss
                    test_kl_loss += kl_loss.item()*len(x)
                else:
                    loss = inn_loss
                test_inn_loss += inn_loss.item()*len(x)
                test_loss += loss.item()*len(x)
        
        # Normalize the losses for printing and plotting and store them also in the corresponding dict
        test_inn_loss /= len(self.test_loader.data)
        test_loss /= len(self.test_loader.data)
               
        self.losses_test['inn'].append(test_inn_loss)
        self.losses_test['total'].append(test_loss)

        if self.model.bayesian:
            test_kl_loss /= len(self.test_loader.data)
            self.losses_test['kl'].append(test_kl_loss)
            
        if self.model.bayesian:
            return test_loss, test_inn_loss, test_kl_loss
                
        return test_loss, test_inn_loss
                  
    def __analyze_logsigs(self):
        """Analyzes the logsigma parameters for a bayesian network.
        """

        logsigs = np.array([])
        
        # Save some parameter values for documentation
        self.close_to_prior.append(0)
        
        # Sigmas only existing for bayesian network
        assert self.model.bayesian
        
        # Initialize the values
        max_bias = 0.0
        max_mu_w = 0.0
        min_logsig2_w = float("inf")
        max_logsig2_w = -float("inf")
        
        # Iterate over all parameters and look at the logsigmas
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                max_bias = max(max_bias, torch.max(torch.abs(param)).item())
            if 'mu_w' in name:
                max_mu_w = max(max_mu_w, torch.max(torch.abs(param)).item())
            if 'logsig2_w' in name:
                self.close_to_prior[-1] += np.sum(np.abs((param - self.logsig2_prior).detach().cpu().numpy()) < 0.01)
                min_logsig2_w = min(min_logsig2_w, torch.min(param).item())
                max_logsig2_w = max(max_logsig2_w, torch.max(param).item())
                logsigs = np.append(logsigs, param.flatten().cpu().detach().numpy())
        
        self.max_logsig.append(max_logsig2_w)
        self.min_logsig.append(min_logsig2_w)
        self.mean_logsig.append(np.mean(logsigs))
        self.median_logsig.append(np.median(logsigs))
        
        return max_bias, max_mu_w, min_logsig2_w, max_logsig2_w
     
    def plot_results(self, epoch, n_samples=None, plot_path=None):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
        self.model.eval()

        # If we are in the final epoch: use more samples!
        if n_samples is not None:
            num_samples = n_samples
            num_rand = 30
        
        elif (not epoch == self.params['n_epochs']):
                num_samples = 10000
                num_rand = 30
        else:
            num_samples = 100000
            num_rand = 30      
        
        
        data, cond_true = self.preprocessor.decode(self.test_loader.data, self.test_loader.cond)

        
        # make plots for the all energies and for the seperate incident energies, as well
        energies = torch.unique(cond_true[:, 0]).to(cond_true.device)
        masks = [torch.ones_like(cond_true[:, 0], dtype=torch.bool, device=cond_true.device)]
        for energy in energies:
            masks.append(cond_true[:, 0] == energy)
            
        for i, mask in enumerate(masks):
            
            # Sample the data using the INN
            if i == 0:
                generated, cond_fake = self.generate(num_samples=num_samples)
            else:
                generated, cond_fake = self.generate(num_samples=num_samples, einc=energies[i-1].item())
            
            # Postprocess the data
            data_post, cond_true_post, layer_boundaries_post = data_util.postprocess(data[mask], cond_true[mask], self.layer_boundaries, self.negative_layers)
            generated_post, cond_fake_post, _ = data_util.postprocess(generated, cond_fake, self.layer_boundaries, self.negative_layers)
            
            # Get the plot paramters
            params = plotting.get_plot_params(layer_boundaries_post, self.coordinates.cpu().numpy(), used_layers=self.params.get("used_calo_layers", None), short=(i!=0))
                
                        
            # Now create the histograms
            if plot_path is None:
                subdir = os.path.join("plots", f'epoch_{epoch:03d}')
                plot_dir = self.doc.get_file(subdir)
            else:
                plot_dir = plot_path
            
            if i == 0:
                name = "summary_all_eincs.pdf"
            else:
                name = f"summary_{energies[i-1].item()}_MeV.pdf"
                
            plotting.plot_all_hist([data_post.cpu().numpy(), generated_post.cpu().numpy()], 
                                [cond_true_post.cpu().numpy(), cond_fake_post.cpu().numpy()], 
                                params, plot_dir=plot_dir, summary_plot=True,
                                summary_plot_name=name, errorbars_true=True,
                                errorbars_fake=True, ncol=5)
        
    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optim = torch.optim.AdamW(
            self.model.params_trainable,
            lr = params.get("lr", 0.0002),
            betas = params.get("betas", [0.9, 0.999]),
            eps = 1e-6,
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
            
        # TODO: Maybe just use step with no decrease?
        elif self.lr_sched_mode == "no_scheduling":
            self.scheduler = None

    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses_test": self.losses_test,
                    "losses_train": self.losses_train,
                    "learning_rates": self.learning_rates,
                    "grads": self.max_grad,
                    "epoch": self.epoch,
                    
                    # Save the logsigma arrays
                    "logsig_min": self.min_logsig,
                    "logsig_max": self.max_logsig,
                    "logsig_mean": self.mean_logsig,
                    "logsig_median": self.median_logsig,
                    "close_to_prior": self.close_to_prior}, self.doc.get_file(f"model{epoch}.pt"))
                    
    def load(self, epoch="", update_offset=True):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        
        if self.VAE_type is not None:
            self.preprocessor.load(epoch)
        
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        if "losses" in state_dicts:
            self.losses_test = state_dicts.get("losses", {})
        elif "losses_test" in state_dicts:
            self.losses_test = state_dicts.get("losses_test", {})
        if "losses_train" in state_dicts:
            self.losses_train = state_dicts.get("losses_train", {})
        self.learning_rates = state_dicts.get("learning_rates", [])
        self.epoch = state_dicts.get("epoch", 0)
        self.max_grad = state_dicts.get("grads", [])
        
        # Load logsigmas, needed for documentation
        self.min_logsig = state_dicts.get("logsig_min",[])
        self.max_logsig = state_dicts.get("logsig_max", [])
        self.mean_logsig = state_dicts.get("logsig_mean", [])
        self.median_logsig = state_dicts.get("logsig_median", [])
        self.close_to_prior = state_dicts.get("close_to_prior", [])

        if update_offset:
            self.epoch_offset = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)

    def _get_incident_energies(self, num_samples):             

        eincs = self.train_loader.cond
        
        energy_values = torch.unique(eincs, return_counts=True)[0]
        
        probabilities = torch.unique(eincs, return_counts=True)[1] /\
            torch.sum(torch.unique(eincs, return_counts=True)[1])
            
        dist = torch.distributions.Categorical(probabilities)
        energies = energy_values[dist.sample((num_samples,1))]
        
        return energies
    
    def _sample_INN(self, energies, batch_size = 1000):     
        self.model.eval() 
        
        with torch.no_grad():
            
            num_samples = len(energies)
            
            # Prepares an "empty" container for the samples
            samples_INN = torch.zeros((num_samples,1,self.num_dim))
            
            # Generate the data in batches according to batch_size
            for batch in range((num_samples+batch_size-1)//batch_size):
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    samples_INN[start:stop] = self.model.sample(1, energies_l).cpu()
                    
            samples_INN = samples_INN[:,0,...]
            
            return samples_INN
     
    def generate(self, num_samples, batch_size = 1000, return_in_training_space=False, einc=None):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for samlpling
        """     
        
        if einc is None:
            energies = self._get_incident_energies(num_samples)           

        else:
            # cast float einc into the shape (num_samples,1) -> Reduplicate along axis 0
            energies = torch.tensor(einc, dtype=torch.get_default_dtype()).repeat(num_samples,1)
            
            
        # 1) INN Part
        samples_INN = self._sample_INN(energies, batch_size)
        
        if return_in_training_space:
            return samples_INN
        
        # 2) VAE Part
        return self.preprocessor.decode(samples_INN, energies, batch_size)

    def latent_samples(self, epoch=None):
        """
            Plot latent space distribution. 

            Parameters:
            epoch (int): current epoch
        """
        self.model.eval()
        with torch.no_grad():
            samples = torch.zeros(self.train_loader.data.shape)
            stop = 0
            for x, c in self.train_loader:
                start = stop
                stop += len(x)
                samples[start:stop] = self.model(x,c)[0].cpu()
                # print(samples)
            samples = samples.numpy()
        plotting.plot_latent(samples, self.doc.basedir, epoch)
    
