import torch
import numpy as np
import time
import os
from torch.optim import Adam

import data_util

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

        train_loader, test_loader = data_utils.get_loaders(
                params.get('data_path'),
                params.get('xml_path'),
                params.get('xml_ptype'),
                params.get('val_frac'),
                params.get('batch_size'),
                device=params.get('device'),
                width_noise=params.get('wodth_noise'),
                ) # list of needed params

        self.doc = doc

    def set_optimizer(self, ):
        self.optimizer_disc = Adam(self.disc_model.trainable_parameters, lr=self.disc_lr)
        self.optimizer_gen = Adam(self.gen_model.trainable_parameters, lr=self.gen_lr)
        
        # add scheduler for both

    def train(self, ):

        for epoch in range(n_epochs):

            for epoch_iter in range(ns_max_iter):

                for disc_iter in range(num_disc_iter):
                    self.optimizer_disc.zero_grad()

                    x_samps = next(disc_train_loader)[0]
                    x_samps.requires_grad = True

                    latent_noise = torch.randn((x_samps.shape[0), self.dim_gen], device=self.device)
                    latent_noise.requires_grad = True

                    pos, neg, x_gen = self.compute_pos_and_neg(latent_noise, x_samps)

                    loss_disc, acc_pos, acc_neg = self.disc_model.loss(pos, neg, epoch=self.epoch, 
                            x_samps=x_samps, x_gen=x_gen)


                    loss_disc.backward()
                    self.optimizer_disc.step()

                    # learning rate scheduling


                for gen_iter in range(num_gen_iter):
                    x_samps = next(gen_train_iter)

                    self.optimizer_gen.zero_grad()

                    z, jac = self.gen_model(x_samps[0], c=x_samps[1])

                    #some relation with the discriminator after 10 epochs
                    if self.epoch >= self.params.get('start_adv', 10):
                        self.gen_model.weight_pot_scheduler(self.epoch)
                        sig = self.predict_discriminator(x_samps, ...).detach()
                        loss_disc = self.gen_model.loss(z, jac, sig)
                    else:
                        loss_disc = self.gen_model.loss(z, jac)

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

    def compute_pos_and_neg(self, latent_noise, x_samps):
        with torch.no_grad():
            x_gen = self.gen_model.sample(latent_noise)
        x_gen.requires_grad = True

        pos = self.disc_model(x_samps)
        neg = self.disc_model(x_gen)
        return pos, neg, x_gen

    def predict_discriminator(self, data, sig, ...):
        ##? doing something with the discriminator





