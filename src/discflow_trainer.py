import torch
import numpy as np
import time
import os
from math import *
from torch.optim import Adam

import data_util
from losses import LatentLoss

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
        
        self.disc_lr = params.get("disc_lr", 1.e-3)
        self.gen_lr = params.get("gen_lr", 1.0e-3)
        self.batch_size = params.get("batch_size", 128)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layer_boundaries = layer_b
        
        #check needed params
        self.latent_loss = LatentLoss(self.params)
        self.dim_gen = train_loader.data.shape[-1]

        self.set_optimizer(len(train_loader))
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
        # add scheduler for both
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
        #params to add
        n_epochs = self.params.get("n_epochs", 1)
        num_disc_iter = self.params.get("n_disc_iter", 1)
        num_gen_iter = self.params.get("n_gen_iter", 1)
        
        ns_max_iter = self.params.get("n_iters_epoch", 
                ceil(len(self.train_loader.data)/self.batch_size/num_gen_iter/num_disc_iter))
      
        self.epoch = 0
        cond = torch.tensor(self.generate_Einc_ds1(energy=None, sample_multiplier=1000), dtype=torch.float)/1e3
        cond = cond.reshape(-1, 1).to(self.device)
        for epoch in range(n_epochs):
            train_inn_loss = 0
            for epoch_iter in range(ns_max_iter):

                for gen_iter in range(num_gen_iter):
                    x_samps = next(iter(self.train_loader))
                        
                    self.optimizer_gen.zero_grad()

                    z, jac = self.gen_model(x_samps[0], c=x_samps[1])

                    #some relation with the discriminator after N epochs
                    if self.epoch >= self.params.get('start_adv', 5):
                        self.latent_loss.weight_pot_scheduler(self.epoch)
                        sig = self.predict_discriminator(disc_x_samps).detach()
                        loss_disc = self.latent_loss.apply(z, jac, sig)
                    else:
                        loss_disc = self.latent_loss.apply(z, jac)

                    loss_inn = 0.0
                    loss_inn += loss_disc

                    loss_inn.backward()
                    self.optimizer_gen.step()

                for disc_iter in range(num_disc_iter):
                    self.optimizer_disc.zero_grad()

                    x_samps, c_samps = next(iter(self.train_loader))
                    disc_x_samps = self.prepare_data_for_discriminator(x_samps, c_samps).to(self.device)

                    disc_x_samps.requires_grad = True
                    x_samps.requires_grad = True

                    #latent_noise = torch.randn((x_samps.shape[0], self.dim_gen), device=self.device)
                    cond_batch = cond[torch.randint(cond.shape[0], size=(self.batch_size,))]

                    #latent_noise.requires_grad = True
                    
                    num_pts = 1
                    pos, neg, x_gen = self.compute_pos_and_neg(num_pts, cond_batch, disc_x_samps)

                    loss_disc, acc_pos, acc_neg = self.disc_model.apply(pos, neg, epoch=self.epoch, 
                            x_samps=x_samps, x_gen=x_gen)

                    loss_disc.backward()
                    self.optimizer_disc.step()

                    # learning rate scheduling
                
                train_inn_loss += loss_inn.item()

                ### plotting for weights

                ### print everything

                # handle learning rates

            self.gen_scheduler.step()

            train_inn_loss /= len(self.train_loader.data)

            #validation
            
            print('')
            print(f'=== epoch {epoch} ===')
            print(f'inn loss (train): {train_inn_loss}')

            self.epoch += 1

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

    #def predict_discriminator(self, data, sig,):
        ##? doing something with the discriminator

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
