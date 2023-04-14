import sys

import os
import numpy as np

import torch

import data_util
from model import CINN, DNN, CVAE
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


class DNNTrainer:
    """ This class is responsible for training and testing the DNN.  """

    def __init__(self, test_trainer, params=None, device=None, doc=None):
        """
            Initializes train_loader, test_loader, DNN, optimizer and scheduler.
            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """
        
        # Save some important parameters
        self.run = 0
        
        # Use the params, device, doc values that are given. If none are passed, use the test_trainer ones.
        if params is not None:
            self.params = params
        else:
            self.params = test_trainer.params
            
        if device is not None:
            self.device = device
        else:
            self.device = test_trainer.device
            
        if doc is not None:
            self.doc = doc
        else:
            self.doc = test_trainer.doc
            
        self.sigmoid_in_BCE = self.params.get("sigmoid_in_BCE", True)
        self.epochs = self.params.get("classifier_n_epochs", 30)
        
        # Select the needed layers:
        layer_index = self.params.get("calo_layer", None) 
        layer_names = ["layer_0", "layer_1", "layer_2"]  # Hardcoded layer names
        
        if layer_index is not None:
            del layer_names[layer_index]
        else:
            layer_names = None
            
        
        # TODO: This is very problem dependent and might be wrong!
        if test_trainer.voxels is None:
            self.postprocessing = True
        else:
            self.postprocessing = False
        
        
        dataloader_train, dataloader_val, dataloader_test = data_util.get_classifier_loaders(test_trainer,
            self.params, self.doc, self.device, drop_layers=layer_names,
            postprocessing=self.postprocessing)
             
        self.train_loader = dataloader_train
        self.val_loader = dataloader_val
        self.test_loader = dataloader_test
        
        # Read the dimensionality from the dataset
        batch_data = next(iter(dataloader_train))
        preprocessed_batch_data, _ = self.__preprocess(batch_data, dtype=torch.get_default_dtype())
        self.num_dim = preprocessed_batch_data.shape[1]


        model = DNN(input_dim=self.num_dim,
                    num_layer=self.params.get("DNN_hidden_layers", 2), 
                    num_hidden=self.params.get("DNN_hidden_neurons", 512), 
                    dropout_probability=self.params.get("DNN_dropout", 0.), 
                    is_classifier=(not self.sigmoid_in_BCE) )
        
        self.model = model.to(self.device)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("classifier_lr", 2e-4))

        self.losses_train = []
        self.losses_test = []
        self.losses_val = []
        self.learning_rates = []
        
        atexit.register(self.clean_up)

    def train(self):
        """ Trains the model. """
        
        print("Start training the model")
        print(f"cuda is used: {next(self.model.parameters()).is_cuda}")
        
        self.run += 1

        best_eval_acc = float('-inf')

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            
            # Initialize the losses
            train_loss = 0
            val_loss = 0
            max_grad = 0.0

            # Train for one epoch
            self.model.train()
            for batch, batch_data in enumerate(self.train_loader):
                
                # Prepare the data of the current batch
                input_vector, target_vector = self.__preprocess(batch_data, dtype=torch.get_default_dtype())
                # print(torch.argwhere(input_vector.isnan()))
                # print(torch.argwhere(target_vector.isnan()))
                output_vector = self.model(input_vector)
                
                # Do the gradient updates
                # print(torch.argwhere(output_vector.isnan()))
                self.optim.zero_grad()
                loss = self.model.loss(output_vector, target_vector.unsqueeze(1))
                loss.backward()
                self.optim.step()

                # Used for plotting
                self.losses_train.append(loss.item())
                
                # Used for printing
                train_loss += loss.item()*len(batch_data)

                # Used for printing 
                for param in self.model.params_trainable:
                    max_grad = max(max_grad, torch.max(torch.abs(param.grad)).item())
                    
                    
                # Store the accuracy on the training set
                if self.sigmoid_in_BCE:
                    pred = torch.round(torch.sigmoid(output_vector.detach()))
                else:
                    pred = torch.round(output_vector.detach())
                target = torch.round(target_vector.detach())
                
                if batch == 0:
                    res_true = target
                    res_pred = pred
                else:
                    res_true = torch.cat((res_true, target), 0)
                    res_pred = torch.cat((res_pred, pred), 0)

            # Check the differences on the validation dataset
            # and save the model with the best val score
            self.model.eval()
            with torch.no_grad():
                for batch, batch_data in enumerate(self.val_loader):
                    
                    input_vector, target_vector = self.__preprocess(batch_data, dtype=torch.get_default_dtype())
                    output_vector = self.model(input_vector)
                    # print(output_vector)
                    
                    pred = output_vector.reshape(-1)
                    target = target_vector
                    
                    # Store the model predictions
                    if batch == 0:
                        result_true = target
                        result_pred = pred
                    else:
                        result_true = torch.cat((result_true, target), 0)
                        result_pred = torch.cat((result_pred, pred), 0)
                    

                # Compute the loss vectorized for all batches of the current epoch
                loss = self.model.loss(result_pred, result_true).cpu().numpy()
                
                # Apply the sigmoid to the prediction, if necessary
                if self.sigmoid_in_BCE:
                    result_pred = torch.sigmoid(result_pred).cpu().numpy()
                else:
                    result_pred = result_pred.cpu().numpy()
                    
                print(result_pred)
                                        
                result_true = result_true.cpu().numpy()
                
                # Compute the accuaracy over the validation set
                eval_acc = accuracy_score(result_true, np.round(result_pred))
                eval_auc = roc_auc_score(result_true, result_pred)
                JSD = - loss + np.log(2.)
  
                
            # Save the model with the best validation loss
            if eval_acc > best_eval_acc:
                self.best_epoch = epoch + 1
                self.save()
                best_eval_acc = eval_acc
      
            # Saved for printing
            val_loss = loss # Must not be normalized, since it is computed over the whole epoch!
            # Normalize (computed per-batch)
            train_loss /= len(self.train_loader.dataset)
            # Saved for plotting
            self.losses_val.append(val_loss)

            # Print the losses of this epoch
            print('')
            print(f'=== run {self.run}, epoch {epoch} ===')
            print(f'loss (train): {train_loss}')
            print(f'loss (validation): {val_loss}')
            print(f'maximum gradient: {max_grad}')
            # Print statements from the old classifier file
            print("Accuracy on training set is", accuracy_score(res_true.cpu(), res_pred.cpu()))
            print("Accuracy on validation set is", eval_acc)
            print("AUC on validation set is", eval_auc)
            print(f"BCE loss of validation set is {loss}, JSD of the two dists is {JSD/np.log(2.)}")
            sys.stdout.flush()

            
            # Plot the monitoring data
            if epoch >= 1:
                sub_path = os.path.join("classifier_test", f"loss_{self.run}.pdf")
                plotting.plot_loss(self.doc.get_file(sub_path), self.losses_train, self.losses_val, skip_epochs=False)
                    
            # Stop training if classifier is already perfect on validation set
            if eval_acc == 1:
                break

    def __preprocess(self, data, dtype=torch.get_default_dtype()):
        """ takes dataloader and returns tensor of
            layer0, layer1, layer2, log10 energy
        """
        device = self.device
        
        if not self.postprocessing:
            return data[0].type(dtype).to(device), data[1].type(dtype).to(device)
        
        # Called for batches in order to prevent ram overflow
        ALPHA = 1e-6

        def logit(x):
            return torch.log(x / (1.0 - x))

        def logit_trafo(x):
            local_x = ALPHA + (1. - 2.*ALPHA) * x
            return logit(local_x)

        threshold=self.params["classifier_threshold"]
        normalize=self.params["classifier_normalize"]
        use_logit=self.params["classifier_use_logit"]

        layer_names = list(data.keys())
        
        # Remove all layers from layer names, that do not correspond
        # to an actual calorimeter layer
        non_calo_layers = ["energy", "overflow", "label"]
        for layer in layer_names:
            # Use naming convention for energies
            if "_E" in layer:
                non_calo_layers.append(layer)
                
        for layer in non_calo_layers:
            layer_names.remove(layer)
        
        layer_data = []
        layer_energy = []
        for layer in layer_names:
            layer_data.append(data[layer])
            layer_energy.append(data[layer + "_E"])
        
        energy = torch.log10(data['energy']*10.).to(device)
        
        # Can be used to remove noise effects
        if threshold:
            for i, layer in enumerate(layer_names):
                layer_data[i] = torch.where(layer_data[i] < 1e-7, torch.zeros_like(layer_data[i]), layer_data[i])

        # Normalize each layer to its total energy
        if normalize:
            for i, layer in enumerate(layer_names):
                layer_data[i] /= (layer_energy[i].reshape(-1, 1, 1) +1e-16)
        
        # Reshape and send to device
        for i, layer in enumerate(layer_names):
            layer_energy[i] = (torch.log10(layer_energy[i].unsqueeze(-1)+1e-8) + 2.).to(device)
            layer_data[i] = layer_data[i].view(layer_data[i].shape[0], -1).to(device)

        # ground truth for the training
        target = data['label'].to(device)
        

        if use_logit:
            for i, layer in enumerate(layer_names):
                layer_data[i] = logit_trafo(layer_data[i]) / 10.
                
                
        # Collect all the tensors and concatenate them afterwards
        final_tensors = []
        for elem in layer_data:
            final_tensors.append(elem)
        final_tensors.append(energy)
        for elem in layer_energy:
            final_tensors.append(elem)

        return torch.cat(final_tensors, 1, ).type(dtype), target.type(dtype)
       
    def __calibrate_classifier(self, calibration_data):
        """ reads in calibration data and performs a calibration with isotonic regression"""
        self.model.eval()
        assert calibration_data is not None, ("Need calibration data for calibration!")
        for batch, data_batch in enumerate(calibration_data):
            input_vector, target_vector = self.__preprocess(data_batch, dtype=torch.get_default_dtype())
            output_vector = self.model(input_vector)
            if self.sigmoid_in_BCE:
                pred = torch.sigmoid(output_vector).reshape(-1)
            else:
                pred = output_vector.reshape(-1)
            target = target_vector
            if batch == 0:
                result_true = target
                result_pred = pred
            else:
                result_true = torch.cat((result_true, target), 0)
                result_pred = torch.cat((result_pred, pred), 0)
        result_true = result_true.cpu().numpy()
        result_pred = result_pred.cpu().numpy()
        iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                        result_true)
        return iso_reg       
    
    def do_classifier_test(self, do_calibration=True):
        # Use the model that worked best on the val set
        self.load()
        
        with torch.no_grad():
            for batch, batch_data in enumerate(self.test_loader):
                    
                input_vector, target_vector = self.__preprocess(batch_data, dtype=torch.get_default_dtype())
                output_vector = self.model(input_vector)
                
                pred = output_vector.reshape(-1)
                target = target_vector
                
                # Store the model predictions
                if batch == 0:
                    result_true = target
                    result_pred = pred
                else:
                    result_true = torch.cat((result_true, target), 0)
                    result_pred = torch.cat((result_pred, pred), 0)
                    
            # Apply the sigmoid to the prediction, if necessary
            if self.sigmoid_in_BCE:
                result_pred = torch.sigmoid(result_pred).cpu().numpy()
            else:
                result_pred = result_pred.cpu().numpy()
                                    
            result_true = result_true.cpu().numpy()
            
            if do_calibration:
                # Isotonic calibration
                calibrator = self.__calibrate_classifier(self.val_loader)
                rescaled_pred = calibrator.predict(result_pred)
            else:
                # TODO:
                # Never reached so far.
                # Not tested so far
                rescaled_pred = result_pred
                
            eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
            print("Rescaled accuracy is", eval_acc)
            eval_auc = roc_auc_score(result_true, rescaled_pred)
            print("rescaled AUC of dataset is", eval_auc)
            prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
            print("rescaled calibration curve:", prob_true, prob_pred)
        
            # Save the test results
            BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
            JSD = - BCE.cpu().numpy() + np.log(2.)
            results = np.array([[eval_acc, eval_auc, JSD/np.log(2.), self.best_epoch]])
            filename = 'summary_DNN.npy'
            sub_path = os.path.join("classifier_test", filename)
            if self.run == 1:
                    np.save(self.doc.get_file(sub_path), results)
            else:
                prev_res = np.load(self.doc.get_file(sub_path),
                                    allow_pickle=True)
                new_res = np.concatenate([prev_res, results])
                np.save(self.doc.get_file(sub_path), new_res)  
            
    def clean_up(self):
        filename = 'summary_DNN.npy'
        sub_path = os.path.join("classifier_test", filename)
        res = np.load(self.doc.get_file(sub_path), allow_pickle=True)        
        
        # Save the averages as txt.file
        filename_average = 'averaged_DNN.txt'
        averaged = np.array([np.mean(res, axis=0), np.std(res, axis=0)])
        sub_path = os.path.join("classifier_test", filename_average)
        np.savetxt(self.doc.get_file(sub_path), averaged)
        
        # Also save it neatly formatted as pdf
        table_filename = "DNN_classifier_results.pdf"
        sub_path = os.path.join("classifier_test", table_filename)
        plotting.plot_average_table(res, self.doc.get_file(sub_path))
      
    def reset_model(self):
        
        model = DNN(input_dim=self.num_dim,
                    num_layer=self.params.get("DNN_hidden_layers", 2), 
                    num_hidden=self.params.get("DNN_hidden_neurons", 512), 
                    dropout_probability=self.params.get("DNN_dropout", 0.), 
                    is_classifier= (not self.sigmoid_in_BCE) )
        self.model = model.to(self.device)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("classifier_lr", 2e-4))

        self.losses_train = []
        self.losses_test = []
        self.losses_val = []
        self.learning_rates = []
      
    def save(self):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        
        sub_path = os.path.join("classifier_test",f"DNN_{self.run}.pt")
        
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses": self.losses_val,
                    "learning_rates": self.learning_rates,
                    "epoch": self.epoch},
                    self.doc.get_file(sub_path))

    def load(self):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        
        sub_path = os.path.join("classifier_test",f"DNN_{self.run}.pt")
        
        state_dicts = torch.load(self.doc.get_file(sub_path), map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])


        self.losses_val = state_dicts.get("losses", {})
        self.learning_rates = state_dicts.get("learning_rates", [])
        self.epoch = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)
        

class VAETrainer:
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
            shuffle=True)
        
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
                 

# OLD: Not updated to the new dataset           
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
        
        self.vae_trainer.load("_best")
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0
        
        # Save the dataloaders ("test" should rather be called validation...)
        self.train_loader, self.test_loader = self.get_loaders()

        # Needed to show the classifier test, that the whole set was used...
        self.voxels = None
        
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
            
            # Append the energy dimensions
            data_train = np.append(data_train, self.vae_trainer.train_loader.cond[:, -6:-3].cpu().numpy(), axis=1)
            data_test = np.append(data_test, self.vae_trainer.test_loader.cond[:, -6:-3].cpu().numpy(), axis=1)
            
            # Create the conditioning data
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

            # Append the energy dimensions
            data_train = np.append(data_train, self.vae_trainer.train_loader.cond[:, -6:-3].cpu().numpy(), axis=1)
            data_test = np.append(data_test, self.vae_trainer.test_loader.cond[:, -6:-3].cpu().numpy(), axis=1)
            
            # Create the conditioning data
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
        
        # Now start the plotting

        generated = self.generate(num_samples=num_samples, return_data=True, save_data=False)
            
        # Now create the no-errorbar histograms
        plotting.plot_all_hist(
            self.doc.basedir,
            self.params['data_path_test'],
            epoch=epoch,
            summary_plot=True, 
            single_plots=False,
            data=generated,
            p_ref=self.params.get("particle_type", "piplus"))  
        
        # Also in the VAE latent space in the last epoch
        
        if epoch == self.params['n_epochs']:
            generated = self.generate(num_samples=num_samples, return_data=True, save_data=False, return_in_training_space=True)
            train_data = self.train_loader.data.cpu().numpy()
            
            bins_1 = plt.hist(generated[:, :-3].flatten(), bins=100)[1]
            plt.close()
            bins_2 = plt.hist(generated[:, -3:].flatten(), bins=100)[1]
            plt.close()
            n = int(np.ceil(generated.shape[1] / 6))

            fig, axs = plt.subplots(n, 6, figsize=(6*6,6*n))
            for i, ax in enumerate(axs.flatten()):
                if i >= generated.shape[1]:
                    break
                
                if i >= generated.shape[1]-3:
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

    def generate(self, num_samples, batch_size = 10000, return_in_training_space=False, return_data=True, save_data=False, postprocessing=True):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for samlpling
        """        
        if postprocessing == False:
            raise NotImplementedError("Cannot do no postprocessing")
        
        self.model.eval()
        self.vae_trainer.model.eval()
        with torch.no_grad():
            # Creates the condition energies uniformly between 1 and 100
            energies = 99.0*torch.rand((num_samples,1)) + 1.0
            
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
            # TODO: Nicer with the VAE generate_from_latent function!


            # For the INN the energy dimensions are part of the training set.
            # For the VAE they are part of the conditioning. So we have to slice them off and
            # append them to the existing E_inc condition.
            
            # Furthermore, we might have to split into mu and sigma parts if we are in this space
            latent_type = self.params.get("latent_type", "pre_sampling")
            
            if (latent_type == "post_sampling") or (latent_type == "only_means"):
                samples_latent = samples_latent[:, :-3]
            elif latent_type == "pre_sampling":
                latent_dim = self.vae_trainer.model.latent_dim
                mu = samples_latent[:, :latent_dim]
                logvar = samples_latent[:, latent_dim:-3]
            
            energy_dims = samples_latent[:, -3:]
            
            # Get layer energies needed as we also append them to the conditions for numerical stability
            u1 = energy_dims[:, [0]]
            u2 = energy_dims[:, [1]]
            u3 = energy_dims[:, [2]]
            e_tot = u1*energies[:, [0]]
            e0 = u2*e_tot
            e1 = u3*(e_tot-e0)
            e2 = e_tot - e0 -e1
            
            # Create the correct condition container
            condition = torch.cat((energies, energy_dims, e0, e1, e2), axis=1)
            
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

            
            # Postprocessing (Reshape the layers and return a dict)
            data = data_util.postprocess(samples, condition)
      
        # Save the sampled data if requested
        if save_data:
            data_util.save_data(
                data = data,
                data_file = self.doc.get_file('samples.hdf5')
            )
            
        # return the sampled data if requested
        if return_data:
            return data

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
    
