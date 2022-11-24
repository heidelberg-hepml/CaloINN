import sys

import os
import numpy as np

import torch

import data_util
from model import CINN, DNN
import plotting
from plotter import Plotter
from classifier import classifier_test
from hdf5_helper import prepare_classifier_datasets

from classifer_data import get_dataloader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve


class INNTrainer:
    """ This class is responsible for training and testing the inn.  """

    def __init__(self, params, device, doc, pretraining=False):
        """
            Initializes train_loader, test_loader, inn, optimizer and scheduler.

            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """

        # Save some important paramters
        self.params = params
        self.device = device
        self.doc = doc
        
        # Pretraining implies a fixed sigma
        self.pretraining = pretraining
        
        # Nedded for printing if the model was loaded
        self.epoch_offset = 0

        # Load the dataloaders
        train_loader, test_loader = data_util.get_loaders(
            data_file_train=params.get('data_path_train'),
            data_file_test=params.get('data_path_test'),
            batch_size=params.get('batch_size'),
            device=device,
            width_noise=params.get("width_noise", 1e-7),
            use_extra_dim=params.get("use_extra_dim", False),
            use_extra_dims=params.get("use_extra_dims", False),
            mask=params.get("mask", 0),
            layer=params.get("calo_layer", None)
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if self.params.get("drop_last", False):
            self.train_loader.drop_last_batch()
            self.test_loader.drop_last_batch()

        self.num_dim = train_loader.data.shape[1]

        data = torch.clone(train_loader.add_noise(train_loader.data))
        cond = torch.clone(train_loader.cond)

        model = CINN(params, data, cond)
        self.model = model.to(device)
        self.set_optimizer(steps_per_epoch=len(train_loader))
        
        if self.pretraining:
            self.model.fix_sigma()

        self.losses_train = {'inn': [], 'kl': [], 'total': []}
        self.losses_test = {'inn': [], 'kl': [], 'total': []}
        self.learning_rates = []
        
        self.max_grad = []
        self.grad_norm = []

    def train(self):
        """ Trains the model. """

        if self.model.bayesian:
            self.model.enable_map()

        self.latent_samples(0)
        N = len(self.train_loader.data)

        for epoch in range(self.epoch_offset+1,self.params['n_epochs']+1):
            self.epoch = epoch
            train_loss = 0
            test_loss = 0
            train_inn_loss = 0
            test_inn_loss = 0
            if self.model.bayesian:
                train_kl_loss = 0
                test_kl_loss = 0
            max_grad = 0.0

            self.model.train()
            for x, c in self.train_loader:
                max_grad_batch = 0.0
                self.optim.zero_grad()
                inn_loss = - torch.mean(self.model.log_prob(x,c))
                if self.model.bayesian:
                    kl_loss = self.model.get_kl() / N
                    
                    loss = inn_loss + self.params.get("kl_weight", 1) * kl_loss
                    self.losses_train['kl'].append(kl_loss.item())
                    train_kl_loss += kl_loss.item()*len(x)
                else:
                    loss = inn_loss
                loss.backward()
                if 'grad_clip' in self.params:
                    # torch.nn.utils.clip_grad_norm_(self.model.params_trainable, self.params['grad_clip'], float('inf'))
                    torch.nn.utils.clip_grad_norm_(self.model.params_trainable, self.params['grad_clip'], 2)
                self.optim.step()

                self.losses_train['inn'].append(inn_loss.item())
                self.losses_train['total'].append(loss.item())
                train_inn_loss += inn_loss.item()*len(x)
                train_loss += loss.item()*len(x)
                if self.scheduler is not None:
                    self.scheduler.step()
                    self.learning_rates.append(self.scheduler.get_last_lr()[0])

                for param in self.model.params_trainable:
                    if param.grad is not None:
                        max_grad_batch = max(max_grad_batch, torch.max(torch.abs(param.grad)).item())
                max_grad = max(max_grad_batch, max_grad)
                self.max_grad.append(max_grad_batch)
                
                if self.params.get("store_grad_norm", False):
                    # Use a code similar to pytorchs clip_grad_norm_:
                    grads = [p.grad for p in self.model.params_trainable if p.grad is not None] 
                    norm_type = float(2) # L2 norm
                    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(self.device) for g in grads]), norm_type)
                    self.grad_norm.append(total_norm.item())
                    

            self.model.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    inn_loss = - torch.mean(self.model.log_prob(x,c))
                    if self.model.bayesian:
                        kl_loss = self.model.get_kl() / N
                        loss = inn_loss + self.params.get("kl_weight", 1) * kl_loss
                        test_kl_loss += kl_loss.item()*len(x)
                    else:
                        loss = inn_loss
                    test_inn_loss += inn_loss.item()*len(x)
                    test_loss += loss.item()*len(x)

            test_inn_loss /= len(self.test_loader.data)
            test_loss /= len(self.test_loader.data)

            train_inn_loss /= len(self.train_loader.data)
            train_loss /= len(self.train_loader.data)

            self.losses_test['inn'].append(test_inn_loss)
            self.losses_test['total'].append(test_loss)

            if self.model.bayesian:
                test_kl_loss /= len(self.test_loader.data)
                train_kl_loss /= len(self.train_loader.data)
                self.losses_test['kl'].append(test_kl_loss)

            if self.model.bayesian:
                max_bias = 0.0
                max_mu_w = 0.0
                min_logsig2_w = 100.0
                max_logsig2_w = -100.0
                for name, param in self.model.named_parameters():
                    if 'bias' in name:
                        max_bias = max(max_bias, torch.max(torch.abs(param)).item())
                    if 'mu_w' in name:
                        max_mu_w = max(max_mu_w, torch.max(torch.abs(param)).item())
                    if 'logsig2_w' in name:
                        min_logsig2_w = min(min_logsig2_w, torch.min(param).item())
                        max_logsig2_w = max(max_logsig2_w, torch.max(param).item())

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

            if epoch >= 1:
                plotting.plot_loss(self.doc.get_file('loss.pdf'), self.losses_train['total'], self.losses_test['total'])
                if self.model.bayesian:
                    plotting.plot_loss(self.doc.get_file('loss_inn.pdf'), self.losses_train['inn'], self.losses_test['inn'])
                    plotting.plot_loss(self.doc.get_file('loss_kl.pdf'), self.losses_train['kl'], self.losses_test['kl'])
                if self.scheduler is not None:
                    plotting.plot_lr(self.doc.get_file('learning_rate.pdf'), self.learning_rates, len(self.train_loader))
                
                plotting.plot_grad(self.doc.get_file('maximum_gradient.pdf'), self.max_grad, len(self.train_loader))
                if self.params.get("store_grad_norm", False):
                    plotting.plot_grad(self.doc.get_file('gradient_norm.pdf'), self.grad_norm, len(self.train_loader))

            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                if epoch % self.params.get("keep_models", self.params["n_epochs"]+1) == 0:
                    self.save(epoch=epoch)
                self.save()
                if (not epoch == self.params['n_epochs']) or self.pretraining:
                    final_plots = False
                    self.generate(10000)
                else:
                    final_plots = True
                    self.generate(100000)

                self.latent_samples(epoch)

                plotting.plot_all_hist(
                    self.doc.basedir,
                    self.params['data_path_test'],
                    mask=self.params.get("mask", 0),
                    calo_layer=self.params.get("calo_layer", None),
                    epoch=epoch,
                    p_ref=self.params.get("particle_type", "piplus"),
                    in_one_file=final_plots)
                
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
        elif self.lr_sched_mode == "no_scheduling":
            self.scheduler = None

    def save(self, epoch=""):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses_test": self.losses_test,
                    "losses_train": self.losses_train,
                    "learning_rates": self.learning_rates,
                    "grads": self.max_grad,
                    "epoch": self.epoch}, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch="", update_offset=False):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
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
        if update_offset:
            self.epoch_offset = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)

    def generate(self, num_samples, batch_size = 10000, return_data=False, save_data=True):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            num_samples (int): Number of samples to generate
            batch_size (int): Batch size for samlpling
        """
        self.model.eval()
        with torch.no_grad():
            energies = 99.0*torch.rand((num_samples,1)) + 1.0
            samples = torch.zeros((num_samples,1,self.num_dim))
            for batch in range((num_samples+batch_size-1)//batch_size):
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    samples[start:stop] = self.model.sample(1, energies_l).cpu()
            samples = samples[:,0,...].cpu().numpy()
            energies = energies.cpu().numpy()
        samples -= self.params.get("width_noise", 1e-7)
        
        data = data_util.postprocess(
                    samples,
                    energies,
                    use_extra_dim=self.params.get("use_extra_dim", False),
                    use_extra_dims=self.params.get("use_extra_dims", False),
                    layer=self.params.get("calo_layer", None)
                )
        
        if save_data:
            data_util.save_data(
                data = data,
                data_file = self.doc.get_file('samples.hdf5')
            )
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
            samples = samples.numpy()
        plotting.plot_latent(samples, self.doc.basedir, epoch)

    def plot_uncertaintys(self, plot_params, num_samples=100000, num_rand=30, batch_size = 10000):
        """
            Plot Bayesian uncertainties for given observables.

            Parameters:
            plot_params (dict): Parameters for the plots to make
            num_samples (int): Number of samples to draw for each instance of the Bayesian parameters
            num_rand (int): Number of random stats to use for the Bayesian parameters
            batch_size (int): Batch size for samlpling
        """
        l_plotter = Plotter(plot_params, self.doc.get_file('plots/uncertaintys'))
        data = data_util.load_data(
            data_file=self.params.get('data_path_test'),
            mask=self.params.get("mask", 0))
        l_plotter.bin_train_data(data)
        self.model.eval()
        for i in range(num_rand):
            self.model.reset_random()
            with torch.no_grad():
                energies = 99.0*torch.rand((num_samples,1)) + 1.0
                samples = torch.zeros((num_samples,1,self.num_dim))
                for batch in range((num_samples+batch_size-1)//batch_size):
                        start = batch_size*batch
                        stop = min(batch_size*(batch+1), num_samples)
                        energies_l = energies[start:stop].to(self.device)
                        samples[start:stop] = self.model.sample(1, energies_l).cpu()
                samples = samples[:,0,...].cpu().numpy()
                energies = energies.cpu().numpy()
            samples -= self.params.get("width_noise", 1e-7)
            data = data_util.postprocess(
                    samples,
                    energies,
                    use_extra_dim=self.params.get("use_extra_dim", False),
                    use_extra_dims=self.params.get("use_extra_dims", False),
                    layer=self.params.get("calo_layer", None)
                )
            l_plotter.update(data)
        l_plotter.plot()

class DNNTrainer:
    """ This class is responsible for training and testing the DNN.  """

    def __init__(self, params, device, doc):
        """
            Initializes train_loader, test_loader, DNN, optimizer and scheduler.
            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """
        
        # TODO: Very ugly
        self.old_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        
        # Save some important parameters
        self.run = 0
        
        self.params = params
        self.device = device
        self.doc = doc
        self.sigmoid_in_BCE = self.params.get("sigmoid_in_BCE", True)
        self.epochs = self.params.get("classifier_n_epochs", 30)
        
        # Generate the test set for the classifier training
        
        train_path, val_path, test_path = prepare_classifier_datasets(
                                    original_dataset=self.params["classification_set"],
                                    generated_dataset=self.doc.get_file('samples.hdf5'),
                                    save_dir=self.doc.basedir)

        
        # TODO: Better use the data_util implementation!
        dataloader_train, dataloader_val, dataloader_test = get_dataloader(
                                                            data_path_train=train_path,
                                                            data_path_test=test_path,
                                                            data_path_val=val_path,
                                                            apply_logit=False,
                                                            device=device,
                                                            batch_size=self.params.get("classifier_batch_size", 1000),
                                                            with_noise=False,
                                                            normed=False,
                                                            normed_layer=False,
                                                            return_label=True)
                
        self.train_loader = dataloader_train
        self.val_loader = dataloader_val
        self.test_loader = dataloader_test
        
        # TODO: Read from data
        self.num_dim = 508


        model = DNN(input_dim=self.num_dim,
                    num_layer=self.params.get("DNN_hidden_layers", 2), 
                    num_hidden=self.params.get("DNN_hidden_neurons", 512), 
                    dropout_probability=self.params.get("DNN_dropout", 0.), 
                    is_classifier=(not self.sigmoid_in_BCE) )
        
        self.model = model.to(device)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.get("classifier_lr", 2e-4))

        self.losses_train = []
        self.losses_test = []
        self.losses_val = []
        self.learning_rates = []

    def train(self):
        """ Trains the model. """
        
        print("Start training the model")
        print(f"cuda is used: {next(self.model.parameters()).is_cuda}")
        
        self.run += 1
        
        # TODO: Consider the dtype bug here!

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
                input_vector, target_vector = self.__preprocess(batch_data)
                output_vector = self.model(input_vector)
                
                # Do the gradient updates
                self.optim.zero_grad()
                loss = self.model.loss(output_vector, target_vector.unsqueeze(1))
                loss.backward()
                # TODO: Gradient clipping?
                self.optim.step()

                # Used for plotting
                self.losses_train.append(loss.item())
                
                # Used for printing
                train_loss += loss.item()*len(batch_data)
                
                # TODO: Maybe use a scheduler?

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
                    
                    input_vector, target_vector = self.__preprocess(batch_data)
                    output_vector = self.model(input_vector)
                    
                    # TODO: Origin of the dtype bug?
                    pred = output_vector.reshape(-1)
                    target = target_vector.double()
                    
                    # Store the model predictions
                    if batch == 0:
                        result_true = target
                        result_pred = pred
                    else:
                        result_true = torch.cat((result_true, target), 0)
                        result_pred = torch.cat((result_pred, pred), 0)
                    

                # Compute the loss vectorized for all batches of the current epoch
                # TODO, NOTE: .CPU Changed
                loss = self.model.loss(result_pred, result_true).cpu().numpy()
                
                # Apply the sigmoid to the prediction, if necessary
                if self.sigmoid_in_BCE:
                    result_pred = torch.sigmoid(result_pred).cpu().numpy()
                else:
                    result_pred = result_pred.cpu().numpy()
                                        
                result_true = result_true.cpu().numpy()
                
                # Compute the accuaracy over the training set
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
            # TODO: Replace .dataset with .data when using other dataloader
            train_loss /= len(self.train_loader.dataset)
            # Saved for plotting
            self.losses_val.append(val_loss)

            # Print the losses of this epoch
            print('')
            print(f'=== epoch {epoch} ===')
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

    def __preprocess(self, data):
        """ takes dataloader and returns tensor of
            layer0, layer1, layer2, log10 energy
        """

        # Called for batches in order to prevent ram overflow
        ALPHA = 1e-6

        def logit(x):
            return torch.log(x / (1.0 - x))

        def logit_trafo(x):
            local_x = ALPHA + (1. - 2.*ALPHA) * x
            return logit(local_x)

        device = self.device
        threshold=self.params["classifier_threshold"]
        normalize=self.params["classifier_normalize"]
        use_logit=self.params["classifier_use_logit"]

        layer0 = data['layer_0']
        layer1 = data['layer_1']
        layer2 = data['layer_2']
        
        energy = torch.log10(data['energy']*10.).to(device)
        
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']

        if threshold:
            layer0 = torch.where(layer0 < 1e-7, torch.zeros_like(layer0), layer0)
            layer1 = torch.where(layer1 < 1e-7, torch.zeros_like(layer1), layer1)
            layer2 = torch.where(layer2 < 1e-7, torch.zeros_like(layer2), layer2)

        if normalize:
            layer0 /= (E0.reshape(-1, 1, 1) +1e-16)
            layer1 /= (E1.reshape(-1, 1, 1) +1e-16)
            layer2 /= (E2.reshape(-1, 1, 1) +1e-16)

        E0 = (torch.log10(E0.unsqueeze(-1)+1e-8) + 2.).to(device)
        E1 = (torch.log10(E1.unsqueeze(-1)+1e-8) + 2.).to(device)
        E2 = (torch.log10(E2.unsqueeze(-1)+1e-8) + 2.).to(device)

        # ground truth for the training
        target = data['label'].to(device)

        layer0 = layer0.view(layer0.shape[0], -1).to(device)
        layer1 = layer1.view(layer1.shape[0], -1).to(device)
        layer2 = layer2.view(layer2.shape[0], -1).to(device)

        if use_logit:
            layer0 = logit_trafo(layer0)/10.
            layer1 = logit_trafo(layer1)/10.
            layer2 = logit_trafo(layer2)/10.

        return torch.cat((layer0, layer1, layer2, energy, E0, E1, E2), 1), target
       
    def __calibrate_classifier(self, calibration_data):
        """ reads in calibration data and performs a calibration with isotonic regression"""
        self.model.eval()
        assert calibration_data is not None, ("Need calibration data for calibration!")
        for batch, data_batch in enumerate(calibration_data):
            input_vector, target_vector = self.__preprocess(data_batch)
            output_vector = self.model(input_vector)
            if self.sigmoid_in_BCE:
                pred = torch.sigmoid(output_vector).reshape(-1)
            else:
                pred = output_vector.reshape(-1)
            # TODO: Another hard-coded dtype!
            target = target_vector.to(torch.float64)
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
                    
                input_vector, target_vector = self.__preprocess(batch_data)
                output_vector = self.model(input_vector)
                
                # TODO: Origin of the dtype bug?
                pred = output_vector.reshape(-1)
                target = target_vector.double()
                
                # Store the model predictions
                if batch == 0:
                    result_true = target
                    result_pred = pred
                else:
                    result_true = torch.cat((result_true, target), 0)
                    result_pred = torch.cat((result_pred, pred), 0)
                    
            # TODO: Remove, if everything works
            # # Compute the loss vectorized for all batches of the current epoch
            # loss = self.model.loss(result_pred, result_true)
            
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
        
        torch.set_default_dtype(self.old_default_dtype)
      
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