import sys

import os
import numpy as np

import torch

import data_util
from model import CINN, CVAE, CCVAE, CAE
import plotting
from plotter import Plotter

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


class VAETrainer:
    def __init__(self, params, device, doc):
        
        self.params = params
        self.device = device
        print(self.device)
        print(params.get("dataset", 1))
        self.doc = doc

        # Load the data  
        self.train_loader, self.test_loader, self.layer_boundaries = data_util.get_loaders(
            filename=params['data_path'],
            particle_type=params['particle_type'],
            val_frac=params["val_frac"],
            batch_size=params['VAE_batch_size'],
            eps=params.get("eps", 1.e-10),
            device=device,
            drop_last=True,
            shuffle=True,
            dataset=params.get("dataset", 1),
            e_inc_index=params.get("e_inc_index", None))
        
        data = self.train_loader.data
        cond = self.train_loader.cond
        
        # Create the VAE
        self.latent_dim = params["VAE_latent_dim"]
        hidden_sizes = params["VAE_hidden_sizes"]
        self.model = CVAE(input = data,
                          cond = cond,
                          latent_dim = self.latent_dim,
                          hidden_sizes = hidden_sizes,
                          layer_boundaries_detector = self.layer_boundaries,
                          particle_type = params['particle_type'],
                          dataset = params.get('dataset', 1),
                          dropout = params.get("VAE_dropout", 0.0),
                          alpha = params.get("alpha", 1.e-6),
                          beta = params.get("VAE_beta", 1.e-5),
                          gamma = params.get("VAE_gamma", 1.e+3),
                          eps = params.get("eps", 1.e-10),
                          noise_width=params.get("VAE_width_noise", None),
                          smearing_self=params.get("VAE_smearing_self", 1.0),
                          smearing_share=params.get("VAE_smearing_share", 0),
                          einc_preprocessing=params.get("VAE_einc_preprocessing", "logit"))
        
        self.model = self.model.to(self.device)
        
        
        # Set the optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("VAE_lr", 1.e-4))
        
        # Configure a possible LR scheduler
        self.set_scheduler()
        
        # Print the model
        print(self.model)
        sys.stdout.flush()
        
        # Needed for documentation (printing & plotting)
        self.losses_train = {'mse': [], 'mse_logit': [],'kl': [], 'total': []}
        self.losses_test = {'mse': [], 'mse_logit': [], 'kl': [], 'total': []}
        self.learning_rates = []
        self.max_grad = []
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0

    def set_scheduler(self):
        
        steps_per_epoch = len(self.train_loader)
        
        if self.params.get("VAE_lr_sched_mode", None) == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size = self.params["lr_decay_epochs"],
                gamma = self.params["lr_decay_factor"],
            )
        elif self.params.get("VAE_lr_sched_mode", None) == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                factor = 0.4,
                patience = 50,
                cooldown = 100,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif self.params.get("VAE_lr_sched_mode", None) == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                self.params.get("VEA_max_lr", self.params["VAE_lr"]*10),
                epochs = self.params.get("opt_epochs") or self.params["n_epochs"],
                steps_per_epoch=steps_per_epoch)
            
        elif self.params.get("VAE_lr_sched_mode", None) is None:
            self.scheduler = None

    def train(self):
        for epoch in range(self.epoch_offset+1, self.params['VAE_n_epochs']+1):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            
            # Initialize the best validation loss
            min_test_loss = np.inf
            
            # Do training and validation for the current epoch
            max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss = self.__train_one_epoch()
            test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss = self.__do_validation()

            # Remember the best model so far
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                self.save("_best")
                
            # Print the data saved for documentation
            self.print_losses(epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss,
                              test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad)
            
            # Plot the losses as well
            if epoch >= 1:
                self.plot_losses()
                
            # If we reach the save interval, create all histograms for the observables,
            # plot the latent distribution and save the model
            if epoch % self.params.get("VAE_keep_models", self.params["VAE_n_epochs"]+1) == 0:
                self.save(epoch=epoch)
            
            if epoch%self.params.get("VAE_save_interval", 100) == 0 or epoch == self.params['VAE_n_epochs']:
                self.save()
                
                self.plot_results(epoch) 
                
            # if epoch == 101:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.9, share_weight=0.025)
            # elif epoch == 201:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.95, share_weight=0.0125)
            # elif epoch == 301:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=1, share_weight=0.)         
         
    def __train_one_epoch(self):
        """Trains the model for one epoch. Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        train_loss = 0
        train_mse_loss = 0
        train_mse_loss_logit = 0
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
            
            # Get the reconstruction loss
            loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
            # Calculate the gradients
            loss.backward()
            
            # Update the parameters
            self.optim.step()

            # Save the losses for documentation
            self.losses_train['mse'].append(mse_loss.item())
            self.losses_train['mse_logit'].append(mse_loss_logit.item())
            self.losses_train['total'].append(loss.item())
            self.losses_train['kl'].append(kl_loss.item())
            
            train_loss += loss.item()*len(x)
            train_mse_loss += mse_loss.item()*len(x)
            train_mse_loss_logit += mse_loss_logit.item()*len(x)
            train_kl_loss += kl_loss.item()*len(x)
            
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
        train_mse_loss /= len(self.train_loader.data)
        train_mse_loss_logit /= len(self.train_loader.data)
        train_kl_loss /= len(self.train_loader.data)
        train_loss /= len(self.train_loader.data)                

        return max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss
                       
    def __do_validation(self):
        """Evaluates the model on the test set.
        Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        test_loss = 0
        test_mse_loss = 0
        test_mse_loss_logit = 0
        test_kl_loss = 0
        
        # Evaluate the model on the test dataset and save the losses
        self.model.eval()
        with torch.no_grad():
            for x, c in self.test_loader:
                
                # Get the reconstruction loss
                loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
                # Save the losses
                test_loss += loss.item() * len(x)
                test_mse_loss += mse_loss.item() * len(x)
                test_mse_loss_logit += mse_loss_logit.item() * len(x)
                test_kl_loss += kl_loss.item() * len(x)
                
        
        # Normalize the losses for printing and plotting and store them also in the corresponding dict
        test_mse_loss /= len(self.test_loader.data)
        test_mse_loss_logit /= len(self.test_loader.data)
        test_loss /= len(self.test_loader.data)
        test_kl_loss /= len(self.test_loader.data)
               
        self.losses_test['mse'].append(test_mse_loss)
        self.losses_test['mse_logit'].append(test_mse_loss_logit)
        self.losses_test['total'].append(test_loss)
        self.losses_test['kl'].append(test_kl_loss)

        return test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss

    def get_reco(self, data, cond, batch_size=10000):
        
        
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
        
    def get_mu_logvar(self, data, cond):
                    
        mu, logvar = self.model.encode(data, cond)
        mu_logvar = torch.cat((mu, logvar), axis=1)
        return mu_logvar
    
    def get_mu(self, data, cond):
                    
        mu, logvar = self.model.encode(data, cond)
        return mu
        
    def get_latent(self, data, cond):
        
        mu, logvar = self.model.encode(x=data, c=cond)
        latent = self.model.reparameterize(mu, logvar)
        return latent
            
    # def generate_from_latent(self, latent, condition, batch_size = 10000):
    #     self.model.eval()
    #     with torch.no_grad():
    #         num_samples = condition.shape[0]

    #         # Prepares an "empty" container for the samples
    #         samples = torch.zeros((num_samples,self.train_loader.data.shape[1]))
    #         for batch in range((num_samples+batch_size-1)//batch_size):
    #             start = batch_size*batch
    #             stop = min(batch_size*(batch+1), num_samples)
    #             condition_l = condition[start:stop].to(self.device)
    #             latent_l = latent[start:stop].to(self.device)
    #             samples[start:stop] = self.model.decode(latent=latent_l, c=condition_l).cpu()

            
    #         # Postprocessing (Reshape the layers and return a dict)
    #         data = data_util.postprocess(samples, condition)

    #         return data      
    
    def plot_results(self, epoch, plot_path=None):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
        
        self.model.eval()
        
        # Generate the reconstructions
        data = self.test_loader.data
        cond = self.test_loader.cond
        generated = self.get_reco(data, cond)
                    
        # Now create the no-errorbar histograms
        if plot_path is None:
            subdir = os.path.join("plots", f'epoch_{epoch:03d}')
            plot_dir = self.doc.get_file(subdir)
        else:
            plot_dir = plot_path
            
        plotting.plot_all_hist(
            data, cond, generated, cond, self.params,
            self.layer_boundaries, plot_dir)

        
        with torch.no_grad():
            mu, logvar = self.model.encode(x=data, c=cond)
            mu0 = mu[:, 0].cpu().numpy()
            mu1 = mu[:, 1].cpu().numpy()
            
            plt.figure(dpi=300)
            plt.plot(mu0, mu1, lw=0, marker=",")
            plt.title(r"$\mu_0$ and $\mu_1$ correlations")
            # plt.xscale("log")
            plt.xlabel(r"$\mu_0$")
            plt.ylabel(r"$\mu_1$")
            # plt.xlim(1.e-9, 5.e1)
            if plot_path is None:
                plt.savefig(self.doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "correlation_plots", "0_1_latent.png")))
            else:
                plt.savefig(os.path.join(plot_path, "0_1_latent.png"))
            plt.close()
      
    def plot_losses(self):
        # Plot the losses
        plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
        plotting.plot_loss(self.doc.get_file('loss_mse_data.pdf'), self.losses_train['mse'], self.losses_test['mse'])
        plotting.plot_loss(self.doc.get_file('loss_mse_logit.pdf'), self.losses_train['mse_logit'], self.losses_test['mse_logit'])
        plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
        
        # Plot the learning rate (if we use a scheduler)
        if self.scheduler is not None:
            plotting.plot_lr(self.doc.get_file('learning_rate.pdf'), self.learning_rates, len(self.train_loader))
        
        # Plot the gradients
        plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))

    def print_losses(self, epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad):
        print('')
        print(f'=== epoch {epoch} ===')
        
        print(f'mse data-loss (train): {train_mse_loss}')
        print(f'mse logit-loss (train): {train_mse_loss_logit}')
        print(f'kl loss (train): {train_kl_loss}')
        print(f'total loss (train): {train_loss}')
        
        print(f'mse data-loss (test): {test_mse_loss}')
        print(f'mse logit-loss (test): {test_mse_loss_logit}')
        print(f'kl loss (test): {test_kl_loss}')
        print(f'total loss (test): {test_loss}')
        
        if self.scheduler is not None:
                print(f'lr: {self.scheduler.get_last_lr()[0]}')

        print(f'maximum gradient: {max_grad}')
        sys.stdout.flush()
         
    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses and the epoch """
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


class CCVAETrainer:
    def __init__(self, params, device, doc):
        
        self.params = params
        self.device = device
        print(self.device)
        self.doc = doc

        # Load the data  
        self.train_loader, self.test_loader, self.layer_boundaries = data_util.get_loaders(
            filename=params['data_path'],
            particle_type=params['particle_type'],
            val_frac=params["val_frac"],
            batch_size=params['VAE_batch_size'],
            eps=params.get("eps", 1.e-10),
            device=device,
            drop_last=False,
            shuffle=True,
            dataset=params.get("dataset", 1))
        
        data = self.train_loader.data
        cond = self.train_loader.cond
        
        # Create the VAE
        self.latent_dim = params["VAE_latent_dim"]
        hidden_sizes = params["VAE_hidden_sizes"]
        self.model = CCVAE(input = data,
                          cond = cond,
                          latent_dim = self.latent_dim,
                          hidden_sizes = hidden_sizes,
                          layer_boundaries_detector = self.layer_boundaries,
                          kernel_size=params["kernel_size"],
                          channel_list=params["channel_list"],
                          padding=params["padding"],
                          particle_type = params['particle_type'],
                          dropout = params.get("VAE_dropout", 0.0),
                          alpha = params.get("alpha", 1.e-6),
                          beta = params.get("VAE_beta", 1.e-5),
                          gamma = params.get("VAE_gamma", 1.e+3),
                          eps = params.get("eps", 1.e-10),
                          noise_width=params.get("VAE_width_noise", None),
                          smearing_self=params.get("VAE_smearing_self", 1.0),
                          smearing_share=params.get("VAE_smearing_share", 0))
        
        self.model = self.model.to(self.device)
        
        
        # Set the optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("VAE_lr", 1.e-4))
        
        # Print the model
        print(self.model)
        sys.stdout.flush()
        
        # Needed for documentation (printing & plotting)
        self.losses_train = {'mse': [], 'mse_logit': [],'kl': [], 'total': []}
        self.losses_test = {'mse': [], 'mse_logit': [], 'kl': [], 'total': []}
        self.learning_rates = []
        self.max_grad = []
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0
        
    def train(self):
        for epoch in range(self.epoch_offset+1, self.params['VAE_n_epochs']+1):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            
            # Initialize the best validation loss
            min_test_loss = np.inf
            
            # Do training and validation for the current epoch
            max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss = self.__train_one_epoch()
            test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss = self.__do_validation()

            # Remember the best model so far
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                self.save("_best")
                
            # Print the data saved for documentation
            self.print_losses(epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss,
                              test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad)
            
            # Plot the losses as well
            if epoch >= 1:
                self.plot_losses()
                
            # If we reach the save interval, create all histograms for the observables,
            # plot the latent distribution and save the model
            if epoch % self.params.get("VAE_keep_models", self.params["VAE_n_epochs"]+1) == 0:
                self.save(epoch=epoch)
            
            if epoch%self.params.get("VAE_save_interval", 100) == 0 or epoch == self.params['VAE_n_epochs']:
                self.save()
                
                self.plot_results(epoch) 
                
            # if epoch == 101:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.9, share_weight=0.025)
            # elif epoch == 201:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.95, share_weight=0.0125)
            # elif epoch == 301:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=1, share_weight=0.)         
         
    def __train_one_epoch(self):
        """Trains the model for one epoch. Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        train_loss = 0
        train_mse_loss = 0
        train_mse_loss_logit = 0
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
            
            # Get the reconstruction loss
            loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
            # Calculate the gradients
            loss.backward()
            
            # Update the parameters
            self.optim.step()

            # Save the losses for documentation
            self.losses_train['mse'].append(mse_loss.item())
            self.losses_train['mse_logit'].append(mse_loss_logit.item())
            self.losses_train['total'].append(loss.item())
            self.losses_train['kl'].append(kl_loss.item())
            
            train_loss += loss.item()*len(x)
            train_mse_loss += mse_loss.item()*len(x)
            train_mse_loss_logit += mse_loss_logit.item()*len(x)
            train_kl_loss += kl_loss.item()*len(x)

            # Save the maximum gradient for documentation
            for param in self.model.parameters():
                if param.grad is not None:
                    max_grad_batch = max(max_grad_batch, torch.max(torch.abs(param.grad)).item())
            max_grad = max(max_grad_batch, max_grad)
            self.max_grad.append(max_grad_batch)
                
        # Normalize the losses to the dataset length and return them.
        # We need a different normalization here compared to the plotting because we summed over the whole epoch!
        train_mse_loss /= len(self.train_loader.data)
        train_mse_loss_logit /= len(self.train_loader.data)
        train_kl_loss /= len(self.train_loader.data)
        train_loss /= len(self.train_loader.data)                

        return max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss
                       
    def __do_validation(self):
        """Evaluates the model on the test set.
        Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        test_loss = 0
        test_mse_loss = 0
        test_mse_loss_logit = 0
        test_kl_loss = 0
        
        # Evaluate the model on the test dataset and save the losses
        self.model.eval()
        with torch.no_grad():
            for x, c in self.test_loader:
                
                # Get the reconstruction loss
                loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
                # Save the losses
                test_loss += loss.item() * len(x)
                test_mse_loss += mse_loss.item() * len(x)
                test_mse_loss_logit += mse_loss_logit.item() * len(x)
                test_kl_loss += kl_loss.item() * len(x)
                
        
        # Normalize the losses for printing and plotting and store them also in the corresponding dict
        test_mse_loss /= len(self.test_loader.data)
        test_mse_loss_logit /= len(self.test_loader.data)
        test_loss /= len(self.test_loader.data)
        test_kl_loss /= len(self.test_loader.data)
               
        self.losses_test['mse'].append(test_mse_loss)
        self.losses_test['mse_logit'].append(test_mse_loss_logit)
        self.losses_test['total'].append(test_loss)
        self.losses_test['kl'].append(test_kl_loss)

        return test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss

    def get_reco(self, data, cond):
        # Do not need batches. If the input fits on the GPU. The output should fit as well. (Consumption goes up by factor of 2)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data, cond)
        
        return reconstructed
        
    def get_mu_logvar(self, data, cond):
                    
        mu, logvar = self.model.encode(data, cond)
        mu_logvar = torch.cat((mu, logvar), axis=1)
        return mu_logvar
    
    def get_mu(self, data, cond):
                    
        mu, logvar = self.model.encode(data, cond)
        return mu
        
    def get_latent(self, data, cond):
        
        mu, logvar = self.model.encode(x=data, c=cond)
        latent = self.model.reparameterize(mu, logvar)
        return latent
    
    def plot_results(self, epoch):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
        
        self.model.eval()
        
        # Generate the reconstructions
        data = self.test_loader.data
        cond = self.test_loader.cond
        generated = self.get_reco(data, cond)
                    
        # Now create the no-errorbar histograms
        subdir = os.path.join("plots", f'epoch_{epoch:03d}')
        plot_dir = self.doc.get_file(subdir)
        plotting.plot_all_hist(
            data, cond, generated, cond, self.params,
            self.layer_boundaries, plot_dir)

        
        with torch.no_grad():
            mu, logvar = self.model.encode(x=data, c=cond)
            mu0 = mu[:, 0].cpu().numpy()
            mu1 = mu[:, 1].cpu().numpy()
            
            plt.figure(dpi=300)
            plt.plot(mu0, mu1, lw=0, marker=",")
            plt.title(r"$\mu_0$ and $\mu_1$ correlations")
            # plt.xscale("log")
            plt.xlabel(r"$\mu_0$")
            plt.ylabel(r"$\mu_1$")
            # plt.xlim(1.e-9, 5.e1)
            plt.savefig(self.doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "correlation_plots", "0_1_latent.png")))
            plt.close()
      
    def plot_losses(self):
        # Plot the losses
        plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
        plotting.plot_loss(self.doc.get_file('loss_mse_data.pdf'), self.losses_train['mse'], self.losses_test['mse'])
        plotting.plot_loss(self.doc.get_file('loss_mse_logit.pdf'), self.losses_train['mse_logit'], self.losses_test['mse_logit'])
        plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
        
        # Plot the gradients
        plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))

    def print_losses(self, epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad):
        print('')
        print(f'=== epoch {epoch} ===')
        
        print(f'mse data-loss (train): {train_mse_loss}')
        print(f'mse logit-loss (train): {train_mse_loss_logit}')
        print(f'kl loss (train): {train_kl_loss}')
        print(f'total loss (train): {train_loss}')
        
        print(f'mse data-loss (test): {test_mse_loss}')
        print(f'mse logit-loss (test): {test_mse_loss_logit}')
        print(f'kl loss (test): {test_kl_loss}')
        print(f'total loss (test): {test_loss}')

        print(f'maximum gradient: {max_grad}')
        sys.stdout.flush()
         
    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses_test": self.losses_test,
                    "losses_train": self.losses_train,
                    "grads": self.max_grad,
                    "epoch": self.epoch,}, 
                   
                   self.doc.get_file(f"model{epoch}.pt"))
                         
    def load(self, epoch="", update_offset=True):
        """ Load the model, its optimizer, losses and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        self.losses_test = state_dicts.get("losses_test", {})
        self.losses_train = state_dicts.get("losses_train", {})
        self.epoch = state_dicts.get("epoch", 0)
        self.max_grad = state_dicts.get("grads", [])
        if update_offset:
            self.epoch_offset = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)
    
    
class AETrainer:
    def __init__(self, params, device, doc):
        
        self.params = params
        self.device = device
        print(self.device)
        self.doc = doc

        # Load the data  
        self.train_loader, self.test_loader, self.layer_boundaries = data_util.get_loaders(
            filename=params['data_path'],
            particle_type=params['particle_type'],
            val_frac=params["val_frac"],
            batch_size=params['VAE_batch_size'],
            eps=params.get("eps", 1.e-10),
            device=device,
            drop_last=False,
            shuffle=True,
            dataset=params.get("dataset", 1))
        
        data = self.train_loader.data
        cond = self.train_loader.cond
        
        # Create the VAE
        self.latent_dim = params["VAE_latent_dim"]
        hidden_sizes = params["VAE_hidden_sizes"]
        self.model = CAE(input = data,
                          cond = cond,
                          latent_dim = self.latent_dim,
                          hidden_sizes = hidden_sizes,
                          layer_boundaries_detector = self.layer_boundaries,
                          particle_type = params['particle_type'],
                          dropout = params.get("VAE_dropout", 0.0),
                          alpha = params.get("alpha", 1.e-6),
                          beta = params.get("VAE_beta", 1.e-5),
                          gamma = params.get("VAE_gamma", 1.e+3),
                          eps = params.get("eps", 1.e-10),
                          noise_width=params.get("VAE_width_noise", None))
        
        self.model = self.model.to(self.device)
        
        
        # Set the optimizer
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("VAE_lr", 1.e-4))
        
        # Print the model
        print(self.model)
        sys.stdout.flush()
        
        # Needed for documentation (printing & plotting)
        self.losses_train = {'mse': [], 'mse_logit': [],'kl': [], 'total': []}
        self.losses_test = {'mse': [], 'mse_logit': [], 'kl': [], 'total': []}
        self.learning_rates = []
        self.max_grad = []
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0
        
    def train(self):
        for epoch in range(self.epoch_offset+1, self.params['VAE_n_epochs']+1):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            
            # Initialize the best validation loss
            min_test_loss = np.inf
            
            # Do training and validation for the current epoch
            max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss = self.__train_one_epoch()
            test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss = self.__do_validation()

            # Remember the best model so far
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                self.save("_best")
                
            # Print the data saved for documentation
            self.print_losses(epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss,
                              test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad)
            
            # Plot the losses as well
            if epoch >= 1:
                self.plot_losses()
                
            # If we reach the save interval, create all histograms for the observables,
            # plot the latent distribution and save the model
            if epoch % self.params.get("VAE_keep_models", self.params["VAE_n_epochs"]+1) == 0:
                self.save(epoch=epoch)
            
            if epoch%self.params.get("VAE_save_interval", 100) == 0 or epoch == self.params['VAE_n_epochs']:
                self.save()
                
                self.plot_results(epoch) 
                
            # if epoch == 101:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.9, share_weight=0.025)
            # elif epoch == 201:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=0.95, share_weight=0.0125)
            # elif epoch == 301:
            #     self.model.update_smearing_matrix(self.train_loader.data, self.train_loader.cond, self_weight=1, share_weight=0.)         
         
    def __train_one_epoch(self):
        """Trains the model for one epoch. Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        train_loss = 0
        train_mse_loss = 0
        train_mse_loss_logit = 0
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
            
            # Get the reconstruction loss
            loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
            # Calculate the gradients
            loss.backward()
            
            # Update the parameters
            self.optim.step()

            # Save the losses for documentation
            self.losses_train['mse'].append(mse_loss.item())
            self.losses_train['mse_logit'].append(mse_loss_logit.item())
            self.losses_train['total'].append(loss.item())
            self.losses_train['kl'].append(kl_loss.item())
            
            train_loss += loss.item()*len(x)
            train_mse_loss += mse_loss.item()*len(x)
            train_mse_loss_logit += mse_loss_logit.item()*len(x)
            train_kl_loss += kl_loss.item()*len(x)

            # Save the maximum gradient for documentation
            for param in self.model.parameters():
                if param.grad is not None:
                    max_grad_batch = max(max_grad_batch, torch.max(torch.abs(param.grad)).item())
            max_grad = max(max_grad_batch, max_grad)
            self.max_grad.append(max_grad_batch)
                
        # Normalize the losses to the dataset length and return them.
        # We need a different normalization here compared to the plotting because we summed over the whole epoch!
        train_mse_loss /= len(self.train_loader.data)
        train_mse_loss_logit /= len(self.train_loader.data)
        train_kl_loss /= len(self.train_loader.data)
        train_loss /= len(self.train_loader.data)                

        return max_grad, train_loss, train_mse_loss, train_mse_loss_logit, train_kl_loss
                       
    def __do_validation(self):
        """Evaluates the model on the test set.
        Saves the losses inplace for plotting and returns the losses that are needed for printing.
        """
        # Initialize the loss values for the documentation
        test_loss = 0
        test_mse_loss = 0
        test_mse_loss_logit = 0
        test_kl_loss = 0
        
        # Evaluate the model on the test dataset and save the losses
        self.model.eval()
        with torch.no_grad():
            for x, c in self.test_loader:
                
                # Get the reconstruction loss
                loss, mse_loss_logit, mse_loss, kl_loss  = self.model.reco_loss(x, c,
                                                                            MAE_logit=self.params.get("VAE_MAE_logit", True),
                                                                            MAE_data=self.params.get("VAE_MAE_data", False),
                                                                            zero_logit=self.params.get("VAE_zero_logit", False),
                                                                            zero_data=self.params.get("VAE_zero_data", False))
                
                # Save the losses
                test_loss += loss.item() * len(x)
                test_mse_loss += mse_loss.item() * len(x)
                test_mse_loss_logit += mse_loss_logit.item() * len(x)
                test_kl_loss += kl_loss.item() * len(x)
                
        
        # Normalize the losses for printing and plotting and store them also in the corresponding dict
        test_mse_loss /= len(self.test_loader.data)
        test_mse_loss_logit /= len(self.test_loader.data)
        test_loss /= len(self.test_loader.data)
        test_kl_loss /= len(self.test_loader.data)
               
        self.losses_test['mse'].append(test_mse_loss)
        self.losses_test['mse_logit'].append(test_mse_loss_logit)
        self.losses_test['total'].append(test_loss)
        self.losses_test['kl'].append(test_kl_loss)

        return test_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss

    def get_reco(self, data, cond):
        # Do not need batches. If the input fits on the GPU. The output should fit as well. (Consumption goes up by factor of 2)
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(data, cond)
        
        return reconstructed
        
    def get_latent(self, data, cond):
        
        latent = self.model.encode(x=data, c=cond)
        return latent
    
    def plot_results(self, epoch):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
        
        self.model.eval()
        
        # Generate the reconstructions
        data = self.test_loader.data
        cond = self.test_loader.cond
        generated = self.get_reco(data, cond)
                    
        # Now create the no-errorbar histograms
        subdir = os.path.join("plots", f'epoch_{epoch:03d}')
        plot_dir = self.doc.get_file(subdir)
        plotting.plot_all_hist(
            data, cond, generated, cond, self.params,
            self.layer_boundaries, plot_dir)
      
    def plot_losses(self):
        # Plot the losses
        plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
        plotting.plot_loss(self.doc.get_file('loss_mse_data.pdf'), self.losses_train['mse'], self.losses_test['mse'])
        plotting.plot_loss(self.doc.get_file('loss_mse_logit.pdf'), self.losses_train['mse_logit'], self.losses_test['mse_logit'])
        plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
        
        # Plot the gradients
        plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))

    def print_losses(self, epoch, train_mse_loss, train_mse_loss_logit, train_kl_loss, train_loss, test_mse_loss, test_mse_loss_logit, test_kl_loss, test_loss, max_grad):
        print('')
        print(f'=== epoch {epoch} ===')
        
        print(f'mse data-loss (train): {train_mse_loss}')
        print(f'mse logit-loss (train): {train_mse_loss_logit}')
        print(f'kl loss (train): {train_kl_loss}')
        print(f'total loss (train): {train_loss}')
        
        print(f'mse data-loss (test): {test_mse_loss}')
        print(f'mse logit-loss (test): {test_mse_loss_logit}')
        print(f'kl loss (test): {test_kl_loss}')
        print(f'total loss (test): {test_loss}')

        print(f'maximum gradient: {max_grad}')
        sys.stdout.flush()
         
    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses_test": self.losses_test,
                    "losses_train": self.losses_train,
                    "grads": self.max_grad,
                    "epoch": self.epoch,}, 
                   
                   self.doc.get_file(f"model{epoch}.pt"))
                         
    def load(self, epoch="", update_offset=True):
        """ Load the model, its optimizer, losses and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])
        self.losses_test = state_dicts.get("losses_test", {})
        self.losses_train = state_dicts.get("losses_train", {})
        self.epoch = state_dicts.get("epoch", 0)
        self.max_grad = state_dicts.get("grads", [])
        if update_offset:
            self.epoch_offset = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)
                                          
       
class ECAETrainer:
    def __init__(self, params, device, doc, vae_dir=None):
        
        # Save some important paramters
        self.params = params
        self.device = device
        self.doc = doc
        
        # Create a VAE trainer, train it and make sure, that the plots of the VAE are put in a different directory
        if vae_dir is None:
            vae_basedir = os.path.join(doc.basedir, "VAE")
            vae_doc = Documenter(params['run_name'], existing_run=True, basedir=vae_basedir, log_name="log_jupyter.txt", read_only=True)
            self.vae_trainer = VAETrainer(params, device, vae_doc)
            print("\n\nStart training of CVAE\n\n")
            self.vae_trainer.train()
            print("\n\nEnd training of CVAE\n\n")
        
        else:
            vae_doc = Documenter(params['run_name'], existing_run=True, basedir=vae_dir, log_name="log_jupyter.txt", read_only=True)
            self.vae_trainer = VAETrainer(params, device, vae_doc)
            
        self.layer_boundaries = self.vae_trainer.layer_boundaries
        self.num_detector_layers = len(self.layer_boundaries) - 1
        
        # TODO: Best or last? Maybe add toggle in params file
        # self.vae_trainer.load("_best")
        self.vae_trainer.load()
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0
        
        # Save the dataloaders ("test" should rather be called validation...)
        self.train_loader, self.test_loader = self.get_loaders()
        
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
              
    def get_loaders(self):
        
        # Makes the lines below much shorter. Used to get the slicing of the extra dims right
        # Cf docstring of "data_util.get_energy_dims"
        n = self.num_detector_layers
        
        batch_size = self.params.get('batch_size')
        
        if not self.params.get("Resample_by_VAE", False):
            
            # TODO: Might actually use shallow copy for the datasets.
            # Could lead to memory problems for larger datasets otherwise
            
            with torch.no_grad():
                latent_type = self.params.get("latent_type", "pre_sampling")
                # Create training and test data:
                if latent_type == "post_sampling":
                    data_train = self.vae_trainer.get_latent(self.vae_trainer.train_loader.data, self.vae_trainer.train_loader.cond).cpu().numpy()
                    data_test = self.vae_trainer.get_latent(self.vae_trainer.test_loader.data, self.vae_trainer.test_loader.cond).cpu().numpy()
                elif latent_type == "pre_sampling":
                    data_train = self.vae_trainer.get_mu_logvar(self.vae_trainer.train_loader.data, self.vae_trainer.train_loader.cond).cpu().numpy()
                    data_test = self.vae_trainer.get_mu_logvar(self.vae_trainer.test_loader.data, self.vae_trainer.test_loader.cond).cpu().numpy()
                # MAP approach
                elif latent_type == "only_means":
                    data_train = self.vae_trainer.get_mu(self.vae_trainer.train_loader.data, self.vae_trainer.train_loader.cond).cpu().numpy()
                    data_test = self.vae_trainer.get_mu(self.vae_trainer.test_loader.data, self.vae_trainer.test_loader.cond).cpu().numpy()
                else:
                    raise KeyError("Don't know this latent type")
            
            # Append the energy dimensions (n is the number of detector layers) -> We do not use the true layer energies anymore.
            # TODO: Watch out for numerical problems later
            data_train = np.append(data_train, self.vae_trainer.train_loader.cond[:, 1:-n].cpu().numpy(), axis=1)
            data_test = np.append(data_test, self.vae_trainer.test_loader.cond[:, 1:-n].cpu().numpy(), axis=1)
            
            # Create the conditioning data (Only the incident energy)
            cond_train = self.vae_trainer.train_loader.cond[:, [0]].cpu().numpy()
            cond_test = self.vae_trainer.test_loader.cond[:, [0]].cpu().numpy()
            
            # Put into the dataloader
            data_train = torch.tensor(data_train, device=self.device, dtype=torch.get_default_dtype())
            cond_train = torch.tensor(cond_train, device=self.device, dtype=torch.get_default_dtype())

            data_test = torch.tensor(data_test, device=self.device, dtype=torch.get_default_dtype())
            cond_test = torch.tensor(cond_test, device=self.device, dtype=torch.get_default_dtype())
            
            # Create the dataloaders
            loader_train = MyDataLoader(data_train, cond_train, batch_size)
            loader_test = MyDataLoader(data_test, cond_test, batch_size)
            
            return loader_train, loader_test
        
        else:
            
            assert self.params.get("latent_type", "pre_sampling") == "post_sampling", "If Resample_by_VAE is used, we need to sample => post_sampling must be true!"
            
            with torch.no_grad():
                # Create training and test data:
                data_train = self.vae_trainer.get_mu_logvar(self.vae_trainer.train_loader.data, self.vae_trainer.train_loader.cond).cpu().numpy()
                data_test = self.vae_trainer.get_mu_logvar(self.vae_trainer.test_loader.data, self.vae_trainer.test_loader.cond).cpu().numpy()

            # Append the energy dimensions (n is the number of detector layers) -> We do not use the true layer energies anymore.
            # TODO: Watch out for numerical problems later
            data_train = np.append(data_train, self.vae_trainer.train_loader.cond[:, 1:-n].cpu().numpy(), axis=1)
            data_test = np.append(data_test, self.vae_trainer.test_loader.cond[:, 1:-n].cpu().numpy(), axis=1)
            
            # Create the conditioning data (Only the incident energy)
            cond_train = self.vae_trainer.train_loader.cond[:, [0]].cpu().numpy()
            cond_test = self.vae_trainer.test_loader.cond[:, [0]].cpu().numpy()
            
            # Put into the dataloader
            data_train = torch.tensor(data_train, device=self.device, dtype=torch.get_default_dtype())
            cond_train = torch.tensor(cond_train, device=self.device, dtype=torch.get_default_dtype())

            data_test = torch.tensor(data_test, device=self.device, dtype=torch.get_default_dtype())
            cond_test = torch.tensor(cond_test, device=self.device, dtype=torch.get_default_dtype())
            
            # Create the dataloaders
            loader_train = MyDataLoader(data_train, cond_train, batch_size)
            loader_test = MyDataLoader(data_test, cond_test, batch_size)
            
            
            loader_train.activate_vae_resampling()
            loader_test.activate_vae_resampling()
            
            return loader_train, loader_test       
               
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
        for epoch in range(self.epoch_offset+1,self.params['n_epochs']+1):
            
            # Save the latest epoch of the training (just the number)
            self.epoch = epoch
            min_test_loss = np.inf
            
            # Do training and validation for the current epoch
            if self.model.bayesian:
                max_grad, train_loss, train_inn_loss, train_kl_loss = self.__train_one_epoch()
                test_loss, test_inn_loss, test_kl_loss = self.__do_validation()
                max_bias, max_mu_w, min_logsig2_w, max_logsig2_w = self.__analyze_logsigs()
            else:       
                max_grad, train_loss, train_inn_loss = self.__train_one_epoch()
                test_loss, test_inn_loss = self.__do_validation()
                
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                self.save("_best")
                
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
     
    def plot_results(self, epoch):
        """Wrapper for the plotting, that calls the functions from plotting.py and plotter.py
        """
        
        # If we are in the final epoch: use more samples!
        if (not epoch == self.params['n_epochs']):
                num_samples = 10000
                num_rand = 30
        else:
            num_samples = 100000
            num_rand = 30
        
        # Now compute the data for the plotting
        generated, cond_fake = self.generate(num_samples=num_samples)
        data = self.vae_trainer.test_loader.data
        cond_true = self.vae_trainer.test_loader.cond
            
        # Now create the no-errorbar histograms
        subdir = os.path.join("plots", f'epoch_{epoch:03d}')
        plot_dir = self.doc.get_file(subdir)
        plotting.plot_all_hist(
            data, cond_true, generated, cond_fake, self.params,
            self.layer_boundaries, plot_dir)
        
        # Also in the VAE latent space in the last epoch
        if epoch == self.params['n_epochs']:
            generated = self.generate(num_samples=num_samples, return_in_training_space=True)
            train_data = self.train_loader.data.cpu().numpy()
            
            bins_1 = plt.hist(generated[:, :-self.num_detector_layers].flatten(), bins=100)[1]
            plt.close()
            bins_2 = plt.hist(generated[:, -self.num_detector_layers:].flatten(), bins=100)[1]
            plt.close()
            n = int(np.ceil(generated.shape[1] / 6))

            fig, axs = plt.subplots(n, 6, figsize=(6*6,6*n))
            for i, ax in enumerate(axs.flatten()):
                if i >= generated.shape[1]:
                    break
                
                if i >= generated.shape[1]-self.num_detector_layers:
                    ax.hist(train_data[:,i], bins=bins_2, density=True)
                    ax.hist(generated[:,i], bins=bins_2, density=True, histtype="step")
                
                else:  
                    ax.hist(train_data[:,i], bins=bins_1, density=True)
                    ax.hist(generated[:,i], bins=bins_1, density=True, histtype="step")
                
            fig.savefig(os.path.join(self.doc.basedir, "plots",  f'epoch_{epoch:03d}',"in_vae_latent.pdf"), bbox_inches='tight', dpi=500)
            plt.close()
        
        # Plot also the errorbar plots if a bayesian model is used
        if self.model.bayesian:
            # TODO: Add uncertainty plots later (How would I do this with a VAE structure ???)
                        
            plotting.plot_overview(self.doc.get_file("overwiev.pdf"),
                                            train_loss=self.losses_train["total"], train_inn_loss=self.losses_train["inn"],
                                            test_loss=self.losses_test["total"], test_inn_loss=self.losses_test["inn"],
                                            learning_rate=self.learning_rates, close_to_prior=self.close_to_prior,
                                            logsigs=[self.max_logsig, self.min_logsig, self.mean_logsig, self.median_logsig],
                                            logsig2_prior=self.logsig2_prior, batches_per_epoch=len(self.train_loader))
            
            plotting.plot_correlation_plots(model = self.model, doc =self.doc, epoch = epoch)

    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optim = torch.optim.AdamW(
            self.model.params_trainable,
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
            
        # TODO: Maybe just use step with no decrease?
        elif self.lr_sched_mode == "no_scheduling":
            self.scheduler = None

    def save(self, epoch="", name=None):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        self.vae_trainer.save(epoch, name)
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
        
        self.vae_trainer.load(epoch)
        
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

    def generate_expanded_cond(self, e_inc, extra_dims):
        
        
        # # Compute the generalized extra dimensions
        # extra_dims = [np.sum(layer_energies_np, axis=1, keepdims=True) / c]

        # for layer_index in range(len(layer_boundaries)-2):
        #     extra_dim = layer_energies_np[..., [layer_index]] / (np.sum(layer_energies_np[..., layer_index:], axis=1, keepdims=True) + eps)
        #     extra_dims.append(extra_dim)
            
        # # Collect all the conditions
        # all_conditions = [c] + extra_dims + layer_energies
        # c = np.concatenate(all_conditions, axis=1)
        
        
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

    def generate(self, num_samples, batch_size = 10000, return_in_training_space=False):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for samlpling
        """     
        # TODO: Won't work if we use high level features as conditions in the VAE!
                
        self.model.eval()
        self.vae_trainer.model.eval()
        
        # TODO: Works only for dataset 1 - photons. Better to implement more general
        energy_values = torch.tensor([
            2.560000e-03, 5.120000e-03, 1.024000e-02, 2.048000e-02,
            4.096000e-02, 8.192000e-02, 1.638400e-01, 3.276800e-01,
            6.553600e-01, 1.310720e+00, 2.621440e+00, 5.242880e+00,
            1.048576e+01, 2.097152e+01, 4.194304e+01])
        
        probabilities = torch.tensor([
            0.08264463, 0.08264463, 0.08264463, 0.08264463, 0.08264463,
            0.08264463, 0.08264463, 0.08264463, 0.08264463, 0.08264463,
            0.08264463, 0.04132231, 0.02479339, 0.01652893, 0.00826446])
        
        e_inc_index = self.params.get("e_inc_index", None)
        if e_inc_index is not None:
            energy_values = energy_values[[e_inc_index]]
            probabilities = torch.tensor([1])
               
        
        with torch.no_grad():
            
            dist = torch.distributions.Categorical(probabilities)
            samples = dist.sample((100,))
            
            # TODO: Make sure to sample the energies correctly
            # Creates the condition energies uniformly between 1 and 100
            energies = energy_values[dist.sample((num_samples,1))]
            
            # 1) INN Part
            # Prepares an "empty" container for the latent samples
            samples_latent = torch.zeros((num_samples,1,self.num_dim))
            
            # Generate the data in batches according to batch_size
            for batch in range((num_samples+batch_size-1)//batch_size):
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    samples_latent[start:stop] = self.model.sample(1, energies_l).cpu()
                    
            samples_latent = samples_latent[:,0,...]
            
            if return_in_training_space:
                return samples_latent
            

            # 2) VAE Part
            
            # For the INN the energy dimensions are part of the training set.
            # For the VAE they are part of the conditioning. So we have to slice them off and
            # append them to the existing E_inc condition.
            
            # Furthermore, we might have to split into mu and sigma parts if we are in this space
            latent_type = self.params.get("latent_type", "pre_sampling")
            
            if (latent_type == "post_sampling") or (latent_type == "only_means"):
                samples_latent = samples_latent[:, :-self.num_detector_layers]
            elif latent_type == "pre_sampling":
                latent_dim = self.vae_trainer.model.latent_dim
                mu = samples_latent[:, :latent_dim]
                logvar = samples_latent[:, latent_dim:-self.num_detector_layers]
            
            extra_dims = samples_latent[:, -self.num_detector_layers:]
            
            
            condition = self.generate_expanded_cond(e_inc=energies, extra_dims=extra_dims)
            
            # Prepares an "empty" container for the samples
            samples = torch.zeros((num_samples,self.vae_trainer.train_loader.data.shape[1]))
            for batch in range((num_samples+batch_size-1)//batch_size):
                start = batch_size*batch
                stop = min(batch_size*(batch+1), num_samples)
                
                condition_l = condition[start:stop].to(self.device)
                
                # Do the reparametrization if needed
                if (latent_type == "post_sampling") or (latent_type == "only_means"):
                    reparametrized_samples_latent_l = samples_latent[start:stop].to(self.device)
                elif latent_type == "pre_sampling":
                    mu_l = mu[start:stop].to(self.device)
                    logvar_l = logvar[start:stop].to(self.device)
                    reparametrized_samples_latent_l = self.vae_trainer.model.reparameterize(mu_l, logvar_l)
                    
                # Samples using the VAE
                samples[start:stop] = self.vae_trainer.model.decode(latent=reparametrized_samples_latent_l, c=condition_l).cpu()

            return  samples, condition

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
    
