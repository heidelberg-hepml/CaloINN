import torch
import numpy as np
import time
import os
import sys
from math import *
from torch.optim import Adam

import data_util
from losses import LatentLoss
import plotting 

import evaluate

class DiscFlow_Trainer():
    ''' trainer for discflow method
    param: parameter file with stuff
    disc_model: class, a classifier torch model with ...
    gen_model: class, a flow model with methods ...
    doc: documenter

    they should have an evaluate loss function method
    '''

    def __init__(self, params, train_loader, test_loader, layer_b, disc_model, gen_model, doc):
        super().__init__()

        self.params = params
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.disc_model = disc_model.to(self.device)
        self.gen_model = gen_model.to(self.device)
        
        self.disc_lr = params.get("disc_lr", 1.e-4)
        self.batch_size = params.get("batch_size", 128)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layer_boundaries = layer_b
        
        #check needed params
        self.latent_loss = LatentLoss(self.params)
        self.dim_gen = train_loader.data.shape[-1]
        self.set_optimizer(steps_per_epoch=len(train_loader))

        # logging
        self.numbers = {'gen_loss': [], 'disc_loss': [], 'disc_acc': [], 'disc_w_pos': [], 'disc_w_neg': []}
        self.test_numbers = {'gen_loss': [], 'disc_loss': [], 'disc_acc': [], 'disc_w_pos': [], 'disc_w_neg': []}
        self.learning_rates = {'generator': [], 'discriminator': []}
        self.doc = doc

    def set_optimizer(self, steps_per_epoch=1):
        params = self.params
        self.optimizer_disc = Adam(self.disc_model.params_trainable, lr=self.disc_lr)
        self.optimizer_gen = torch.optim.AdamW(
                self.gen_model.params_trainable,
                lr = params.get("lr", 0.0002),
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )

        self.lr_sched_mode = params.get("lr_scheduler", "one_cycle_lr")
        self.disc_lr_sched_mode = params.get("disc_lr_scheduler", "reduce_on_plateau")

        # add scheduler for both
        if self.disc_lr_sched_mode == "reduce_on_plateau":
            self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_disc,
                factor = 0.8,
                patience = 50,
                cooldown = 100,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose = True
                )
        else:
            raise NotImplementedError()

        if self.lr_sched_mode == "one_cycle_lr":
            self.gen_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer_gen,
                params.get("max_lr", params["lr"]*10),
                epochs = params.get("cycle_epochs") or params["n_epochs"],
                steps_per_epoch=steps_per_epoch,
                )
        else:
            raise NotImplementedError()
     

    def train(self, ):
        n_epochs = self.params.get("n_epochs", 1)
        num_disc_iter = self.params.get("n_disc_iter", 1)
        num_gen_iter = self.params.get("n_gen_iter", 1)
        
        ns_max_iter = self.params.get("n_iters_epoch", 
                ceil(len(self.train_loader.data)/self.batch_size/num_gen_iter/num_disc_iter))
      
        self.epoch = 0
        cond = torch.tensor(self.generate_Einc_ds1(energy=self.params.get('single_energy', None), sample_multiplier=1000), dtype=torch.float)/1e3
        cond = cond.reshape(-1, 1).to(self.device)
        for epoch in range(n_epochs):
            train_inn_loss = 0
            train_disc_loss = 0
            train_acc_mean = 0
            train_wpos_mean = 0
            train_wneg_mean = 0

            test_inn_loss = 0

            self.gen_model.train()
            self.disc_model.train()
            for epoch_iter in range(ns_max_iter):

                for gen_iter in range(num_gen_iter):
                    x_samps = next(iter(self.train_loader))
                        
                    self.optimizer_gen.zero_grad()

                    z, jac = self.gen_model(x_samps[0], c=x_samps[1])

                    # start reweighting the loss after 'start_adv' epochs
                    if self.epoch >= self.params.get('start_adv', 5):
                        self.latent_loss.weight_pot_scheduler(self.epoch)
                        sig = self.predict_discriminator(disc_x_samps).detach() # disc_x_samps??
                        loss_disc = self.latent_loss.apply(z, jac, sig)
                    else:
                        loss_disc = self.latent_loss.apply(z, jac)

                    loss_inn = loss_disc

                    loss_inn.backward()
                    self.optimizer_gen.step()
                    self.gen_scheduler.step()
                    self.learning_rates['generator'].append(self.optimizer_gen.param_groups[0]['lr'])
                    
                    train_inn_loss += loss_inn.item()*len(x_samps[0])
                    self.numbers['gen_loss'].append(loss_inn.item())
                for disc_iter in range(num_disc_iter):
                    self.optimizer_disc.zero_grad()

                    x_samps, c_samps = next(iter(self.train_loader))
                    disc_x_samps = self.prepare_data_for_discriminator(x_samps, c_samps).to(self.device)

                    disc_x_samps.requires_grad = True
                    x_samps.requires_grad = True

                    cond_batch = cond[torch.randint(cond.shape[0], size=(self.batch_size,))]

                    num_pts = 1
                    pos, neg, x_gen = self.compute_pos_and_neg(num_pts, cond_batch, disc_x_samps)

                    loss_disc = 0.0
                    loss_disc, acc_pos, acc_neg = self.disc_model.apply(pos, neg, epoch=self.epoch, 
                            x_samps=x_samps, x_gen=x_gen)

                    loss_disc.backward()
                    self.learning_rates['discriminator'].append(self.optimizer_disc.param_groups[0]['lr'])

                    weights_pos = (torch.sigmoid(pos)/(1-torch.sigmoid(pos)))
                    weights_neg = (torch.sigmoid(neg)/(1-torch.sigmoid(neg)))

                    train_disc_loss += loss_disc.item()*len(x_samps)
                    train_acc_mean += (acc_pos.mean()+acc_neg.mean())/2*len(x_samps)
                    train_wpos_mean += weights_pos.mean().item()*len(x_samps)
                    train_wneg_mean += weights_neg.mean().item()*len(x_samps)

                    self.optimizer_disc.step()
            self.disc_scheduler.step(loss_disc.item())
                    # learning rate scheduling
                
            ### plotting for weights of test data
            self.disc_model.eval()
            test_weights_pos = []
            test_weights_neg = []
            with torch.no_grad():
                for x, c in self.test_loader:
                    test_x = self.prepare_data_for_discriminator(x, c).to(self.device)
                    test_cond_batch =  cond[torch.randint(cond.shape[0], size=(self.batch_size,))]

                    test_pos, test_neg, _ = self.compute_pos_and_neg(1, test_cond_batch, test_x)
                    test_weights_pos.append(torch.sigmoid(test_pos)/(1-torch.sigmoid(test_pos)))
                    test_weights_neg.append(torch.sigmoid(test_neg)/(1-torch.sigmoid(test_neg)))
            test_weights_pos = torch.cat(test_weights_pos, dim=0).detach().cpu().numpy()
            test_weights_neg = torch.cat(test_weights_neg, dim=0).detach().cpu().numpy()
            plotting.plot_weights(test_weights_pos, test_weights_neg, results_dir=self.doc.basedir, epoch=self.epoch)

            #update losses
            train_inn_loss /= len(self.train_loader.data)
            train_disc_loss /= len(self.train_loader.data)
            train_acc_mean /= len(self.train_loader.data)
            train_wpos_mean /= len(self.train_loader.data)
            train_wneg_mean /= len(self.train_loader.data)

            #self.numbers['gen_loss'].append(train_inn_loss)
            self.numbers['disc_loss'].append(train_disc_loss)
            self.numbers['disc_acc'].append(train_acc_mean)
            self.numbers['disc_w_pos'].append(train_wpos_mean)
            self.numbers['disc_w_neg'].append(train_wneg_mean)

            self.epoch += 1
            
            #validation
            self.gen_model.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    inn_loss = - torch.mean(self.gen_model.log_prob(x,c))
                    loss = inn_loss
                    test_inn_loss += inn_loss.item()*len(x)

            test_inn_loss /= len(self.test_loader.data)

            #print everything 
            print('')
            print(f'=== epoch {epoch} ===')
            print(f'inn loss (train): {train_inn_loss}')
            print(f'disc loss (train): {train_disc_loss}')
            print(f'disc accuracy (train): {train_acc_mean}')
            print(f'disc pos weights (train): {train_wpos_mean}')
            print(f'disc neg weights (train): {train_wneg_mean}')

            print(f'inn loss (test): {test_inn_loss}')
            sys.stdout.flush()

            self.test_numbers['gen_loss'].append(test_inn_loss)
            # Plotting loss function and learning rate
            if epoch >= 1:
                plotting.plot_loss(self.doc.get_file('gen_loss.png'), self.numbers['gen_loss'], self.test_numbers['gen_loss'])
                plotting.plot_loss(self.doc.get_file('disc_loss.png'), self.numbers['disc_loss'], self.numbers['disc_loss'])
                #plotting.plot_loss()
            plotting.plot_lr(self.doc.get_file('learning_rate_gen.png'), self.learning_rates['generator'], len(self.train_loader))
            plotting.plot_lr(self.doc.get_file('learning_rate_disc.png'), self.learning_rates['discriminator'], 1)
            
            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                self.save(str(epoch))
                gen_data = self.generate()

                # move everything to CaloChallenge evaluate.py
                if epoch < 70:
                    evaluate.main(f"-i {self.doc.basedir}/samples.hdf5 -r {self.params['val_data_path']} -m no-cls -d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/eval/{epoch}/ --cut 1.0e-3 --energy {self.params['single_energy']}".split())
                else:
                    evaluate.main(f"-i {self.doc.basedir}/samples.hdf5 -r {self.params['val_data_path']} -m all -d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/eval/{epoch}/ --cut 1.0e-3 --energy {self.params['single_energy']}".split())

        print("DiscFlow madness is over")

    def predict_discriminator(self, data):
        with torch.no_grad():
            labels = torch.zeros(len(data), device=data.device)
            batch_size = self.batch_size # set batch size
            for i in range(ceil(len(data) / batch_size)):
                batch = data[i*batch_size:(i+1)*batch_size]
                labels[i*batch_size:(i+1)*batch_size] = self.disc_model( batch , sig=True).squeeze()
            return labels

    def compute_pos_and_neg(self, num_pts, cond, disc_x_samps):
        with torch.no_grad():
            x_gen = self.gen_model.sample(num_pts, cond)
        x_gen = x_gen[:,0]

        disc_x_gen = self.prepare_data_for_discriminator(x_gen, cond).to(self.device)
        disc_x_gen.requires_grad = True

        pos = self.disc_model(disc_x_samps)
        neg = self.disc_model(disc_x_gen)
        return pos, neg, x_gen

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1 
			sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
			and 5, 3, 2, 1 times sample multiplier for the highest energies
		
        """
        ret = np.logspace(8,18,11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array([*ret, *np.tile(2.**19, 5), *np.tile(2.**20, 3), *np.tile(2.**21, 2), *np.tile(2.**22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret==energy]
        np.random.shuffle(ret)
        return ret
    
    def generate(self, batch_size = 10000):
        """
            generate new data using the modle and storing them to a file in the run folder.

            Parameters:
            batch_size (int): Batch size for samlpling
        """
        self.gen_model.eval()
        with torch.no_grad():
            energies = (torch.tensor(self.generate_Einc_ds1(energy=self.params.get('single_energy', None), sample_multiplier=1000), dtype=torch.float)/1e3).reshape(-1, 1)
            samples = torch.zeros((energies.shape[0],1,self.dim_gen))
            num_samples = energies.shape[0]
            times = []
            for batch in range((num_samples+batch_size-1)//batch_size):
                    #self.model.reset_random()
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    t1 = time.time()
                    samples[start:stop] = self.gen_model.sample(1, energies_l)
                    t_diff = time.time() - t1
                    times.append( t_diff/(stop-start) )
            self.avg_gen_time = np.array(times).mean() 
            samples = samples[:,0,...].cpu().numpy()
            energies = energies.cpu().numpy()
        samples -= self.params.get("width_noise", 1e-7)
        data_util.save_data(
            data = data_util.postprocess(
                samples,
                energies,
                layer_boundaries=self.layer_boundaries,
                threshold=self.params.get("width_noise", 1e-7),
                quantiles=self.params.get("width_noise", 1e-7),
            ),
            filename = self.doc.get_file('samples.hdf5')
        )
        data = data_util.postprocess(
                samples,
                energies,
                layer_boundaries=self.layer_boundaries,
                threshold=self.params.get("width_noise", 1e-7),
                quantiles=self.params.get("width_noise", 1e-7),
            )
        return data

    def prepare_data_for_discriminator(self, x_samp, c_samp):
        x_samp_np = np.copy(x_samp.cpu().numpy())
        c_samp_np = np.copy(c_samp.cpu().numpy())

        x_samp_np[x_samp_np < self.params.get("width_noise", 1e-7)] = 0.0
        input_data = data_util.unnormalize_layers(
                x_samp_np,
                c_samp_np,
                self.layer_boundaries,
                )
        input_data /= c_samp_np
        input_data = np.concatenate((input_data, np.log10(c_samp_np)), axis=-1)
        return torch.tensor(input_data, dtype=x_samp.dtype)

    def save(self, epoch=""):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({#"opt": self.optim.state_dict(),
                    "gen_net": self.gen_model.state_dict(),
                    "disc_net": self.disc_model.state_dict(),
                    #"losses": self.losses_test,
                    #"learning_rates": self.learning_rates,
                    }#"epoch": self.epoch}
                    , self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.gen_model.load_state_dict(state_dicts["gen_net"])
        self.disc_model.load_state_dict(state_dicts["disc_net"])

        #self.losses_test = state_dicts.get("losses", {})
        #self.learning_rates = state_dicts.get("learning_rates", [])
        #self.epoch = state_dicts.get("epoch", 0)
        #self.optim.load_state_dict(state_dicts["opt"])
        self.gen_model.to(self.device)
        self.disc_model.to(self.device)


