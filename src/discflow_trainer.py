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

    def __init__(self, params, disc_model, gen_model, doc):
        super().__init__()

        self.params = params
        self.disc_model = disc_model
        self.gen_model = gen_model
        
        self.disc_lr = params.get("disc_lr", 1.e-3)
        self.gen_lr = params.get("gen_lr", 1.0e-3)

        train_loader, test_loader, layer_b = data_util.get_loaders(
                params.get('data_path'),
                params.get('xml_path'),
                params.get('xml_ptype'),
                params.get('val_frac'),
                params.get('batch_size'),
                device=params.get('device'),
                width_noise=params.get('width_noise'),
                ) # list of needed params
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layer_boundaries = layer_b
        
        #check needed params
        self.latent_loss = LatentLoss(self.params)
        self.dim_gen = train_loader.data.shape[-1]

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.doc = doc

    def set_optimizer(self, ):
        self.optimizer_disc = Adam(self.disc_model.params_trainable, lr=self.disc_lr)
        self.optimizer_gen = Adam(self.gen_model.params_trainable, lr=self.gen_lr)
        
        # add scheduler for both

    def train(self, ):
        #params to add
        n_epochs = 1
        ns_max_iter = 1
        num_disc_iter = 100
        num_gen_iter = 1
        
        self.epoch = 0
        for epoch in range(n_epochs):

            for epoch_iter in range(ns_max_iter):

                for disc_iter in range(num_disc_iter):
                    self.optimizer_disc.zero_grad()

                    x_samps = next(iter(self.train_loader))[0]
                    x_samps.requires_grad = True

                    #latent_noise = torch.randn((x_samps.shape[0], self.dim_gen), device=self.device)
                    cond = 1 + torch.rand((self.dim_gen), device=self.device)*99  # add conditions for latent_noise
                    cond = cond.reshape(-1, 1)
                    #latent_noise.requires_grad = True
                    
                    num_pts = 1
                    pos, neg, x_gen = self.compute_pos_and_neg(num_pts, cond, x_samps)
                        
                    loss_disc, acc_pos, acc_neg = self.disc_model.apply(pos, neg, epoch=self.epoch, 
                            x_samps=x_samps, x_gen=x_gen)

                    print(acc_pos, acc_neg)
                    loss_disc.backward()
                    self.optimizer_disc.step()

                    # learning rate scheduling


                for gen_iter in range(num_gen_iter):
                    x_samps = next(iter(self.train_loader))

                    self.optimizer_gen.zero_grad()

                    z, jac = self.gen_model(x_samps[0], c=x_samps[1])

                    #some relation with the discriminator after 10 epochs
                    if True: #self.epoch >= self.params.get('start_adv', 10):
                        self.latent_loss.weight_pot_scheduler(self.epoch)
                        sig = self.predict_discriminator(x_samps[0]).detach()
                        loss_disc = self.latent_loss.apply(z, jac, sig)
                    else:
                        loss_disc = self.latent_loss.apply(z, jac)

                    loss_inn = 0.0
                    loss_inn += loss_disc

                    loss_inn.backward()
                    self.optimizer_gen.step()

                    # learning rate scheduling for generator



                ### plotting for weights

                ### print everything

                # handle learning rates


            self.epoch += 1

            #validation

            print("DiscFlow madness is over")

    def predict_discriminator(self, data):
        with torch.no_grad():
            labels = torch.zeros(len(data), device=data.device)
            batch_size = 128 # set batch size
            for i in range(ceil(len(data) / batch_size)):
                batch = data[i*batch_size:(i+1)*batch_size]
                labels[i*batch_size:(i+1)*batch_size] = self.disc_model( batch ).squeeze()
            return labels

    def compute_pos_and_neg(self, num_pts, cond, x_samps):
        with torch.no_grad():
            x_gen = self.gen_model.sample(num_pts, cond)
        x_gen.requires_grad = True

        pos = self.disc_model(x_samps)
        neg = self.disc_model(x_gen)
        return pos, neg, x_gen

    #def predict_discriminator(self, data, sig,):
        ##? doing something with the discriminator





