import sys

import os
import numpy as np

import torch

import data_util
from model import CINN
import plotting
from plotter import Plotter
from classifier import classifier_test
from hdf5_helper import prepare_classifier_datasets

class Trainer:
    """ This class is responsible for training and testing the model.  """

    def __init__(self, params, device, doc):
        """
            Initializes train_loader, test_loader, model, optimizer and scheduler.

            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """

        self.params = params
        self.device = device
        self.doc = doc

        # Actually split is now without function!
        train_loader, test_loader = data_util.get_loaders(
            params.get('data_path_train'),
            params.get('data_path_test'),
            params.get('batch_size'),
            params.get('train_split', 0.8),
            device,
            params.get("width_noise", 1e-7),
            params.get("use_extra_dim", False),
            params.get("use_extra_dims", False),
            params.get("mask", 0),
            params.get("calo_layer", None)
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_dim = train_loader.data.shape[1]

        data = torch.clone(train_loader.add_noise(train_loader.data))
        cond = torch.clone(train_loader.cond)

        model = CINN(params, data, cond)
        self.model = model.to(device)
        self.set_optimizer(steps_per_epoch=len(train_loader))

        self.losses_train = {'inn': [], 'kl': [], 'total': []}
        self.losses_test = {'inn': [], 'kl': [], 'total': []}
        self.learning_rates = []

    def train(self):
        """ Trains the model. """

        if self.model.bayesian:
            self.model.enable_map()

        self.latent_samples(0)
        N = len(self.train_loader.data)

        for epoch in range(1,self.params['n_epochs']+1):
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
                self.optim.zero_grad()
                inn_loss = - torch.mean(self.model.log_prob(x,c))
                if self.model.bayesian:
                    kl_loss = self.model.get_kl() / N
                    loss = inn_loss + kl_loss
                    self.losses_train['kl'].append(kl_loss.item())
                    train_kl_loss += kl_loss.item()*len(x)
                else:
                    loss = inn_loss
                loss.backward()
                if 'grad_clip' in self.params:
                    torch.nn.utils.clip_grad_norm_(self.model.params_trainable, self.params['grad_clip'], float('inf'))
                self.optim.step()

                self.losses_train['inn'].append(inn_loss.item())
                self.losses_train['total'].append(loss.item())
                train_inn_loss += inn_loss.item()*len(x)
                train_loss += loss.item()*len(x)
                self.scheduler.step()
                self.learning_rates.append(self.scheduler.get_last_lr()[0])

                for param in self.model.params_trainable:
                    max_grad = max(max_grad, torch.max(torch.abs(param.grad)).item())

            self.model.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    inn_loss = - torch.mean(self.model.log_prob(x,c))
                    if self.model.bayesian:
                        kl_loss = self.model.get_kl() / N
                        loss = inn_loss + kl_loss
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
            print(f'lr: {self.scheduler.get_last_lr()[0]}')
            if self.model.bayesian:
                print(f'maximum bias: {max_bias}')
                print(f'maximum mu_w: {max_mu_w}')
                print(f'minimum logsig2_w: {min_logsig2_w}')
                print(f'maximum logsig2_w: {max_logsig2_w}')
            print(f'maximum gradient: {max_grad}')
            sys.stdout.flush()

            if epoch >= 1:
                plotting.plot_loss(self.doc.get_file('loss.png'), self.losses_train['total'], self.losses_test['total'])
                if self.model.bayesian:
                    plotting.plot_loss(self.doc.get_file('loss_inn.png'), self.losses_train['inn'], self.losses_test['inn'])
                    plotting.plot_loss(self.doc.get_file('loss_kl.png'), self.losses_train['kl'], self.losses_test['kl'])
                plotting.plot_lr(self.doc.get_file('learning_rate.png'), self.learning_rates, len(self.train_loader))

            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                self.save()
                if not epoch == self.params['n_epochs']:
                    final_plots = False
                    self.generate(10000)
                else:
                    final_plots = True
                    self.generate(100000)

                self.latent_samples(epoch)

                # TODO: Check again (uses the test data for plotting)
                plotting.plot_all_hist(
                    self.doc.basedir,
                    self.params['data_path_test'],
                    mask=self.params.get("mask", 0),
                    calo_layer=self.params.get("calo_layer", None),
                    epoch=epoch,
                    p_ref=self.params.get("particle_type", "piplus"),
                    in_one_file=final_plots)

            # Classifier test
            # TODO: Better implement in the documenter class
            if (epoch == self.params['n_epochs']) and self.params.get('do_classifier_test', False):
                self.start_classifier_test()

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

    def save(self, epoch=""):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses": self.losses_test,
                    "learning_rates": self.learning_rates,
                    "epoch": self.epoch}, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        self.losses_test = state_dicts.get("losses", {})
        self.learning_rates = state_dicts.get("learning_rates", [])
        self.epoch = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)

    def generate(self, num_samples, batch_size = 10000):
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
        data_util.save_data(
            data = data_util.postprocess(
                samples,
                energies,
                use_extra_dim=self.params.get("use_extra_dim", False),
                use_extra_dims=self.params.get("use_extra_dims", False),
                layer=self.params.get("calo_layer", None)
            ),
            data_file = self.doc.get_file('samples.hdf5')
        )

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
        # TODO: Check again (uses the test data for plotting)
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
        
    def start_classifier_test(self):
        
        # Just use the 100000 data points generated anyway
        # self.generate(100000)
        
        # TODO: use train or test set here?
        train_path, val_path, test_path = prepare_classifier_datasets(
                                    original_dataset=self.params["classification_set"],
                                    generated_dataset=self.doc.get_file('samples.hdf5'),
                                    save_dir=self.doc.basedir)
        
        save_dir = os.path.join(self.doc.basedir, "classifier_test")
        
        
        # TODO: Input dimension is hard coded
        number_of_runs = self.params["classifier_runs"]
        for run_number in range(number_of_runs):
            classifier_test(input_dim=508,
                            device=self.device,
                            data_path_train=train_path,
                            data_path_val=val_path,
                            data_path_test=test_path,
                            save_dir=save_dir,
                            threshold=self.params["classifier_threshold"],
                            normalize=self.params["classifier_normalize"],
                            use_logit=self.params["classifier_use_logit"],
                            sigmoid_in_BCE=self.params.get("sigmoid_in_BCE", True),
                            lr=self.params.get("classifier_lr", 2e-4),
                            n_epochs=self.params.get("classifier_n_epochs", 30),
                            batch_size=self.params.get("classifier_batch_size", 1000),
                            load=False,
                            num_layer=self.params.get("DNN_hidden_layers", 2),
                            num_hidden=self.params.get("DNN_hidden_neurons", 512),
                            dropout_probability=self.params.get("DNN_dropout", 0.),
                            run_number=run_number,
                            modes=self.params.get("modes", ["DNN", "CNN"]))
        
        load = False
        for mode in self.params.get("modes", ["DNN", "CNN"]):
            filename = 'summary_'+('loaded_' if load else '')+mode+'.npy'
            res = np.load(os.path.join(save_dir, filename), allow_pickle=True)            
            
            # Save the averages as txt.file
            filename_average = 'averaged_'+('loaded_' if load else '')+mode+'.txt'
            averaged = np.array([np.mean(res, axis=0), np.std(res, axis=0)])
            np.savetxt(os.path.join(save_dir, filename_average), averaged)
            
            # Also save it neatly formatted as pdf
            table_filename = os.path.join(save_dir, mode + "_classifier_results.pdf")
            plotting.plot_average_table(res, table_filename)

            