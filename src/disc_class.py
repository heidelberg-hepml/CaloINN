from FrEIA.framework import *
from FrEIA.modules import *
import numpy as np
import torch
import torch.nn as nn
import time
from math import ceil, floor
import os

class DNN(torch.nn.Module):
    """ NN for vanilla classifier """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()
        
        self.BCE = nn.BCEWithLogitsLoss()
        self.dpo = dropout_probability
        
        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

        self.params_trainable = list(filter(
            lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, sig=False):
        x = self.layers(x)
        if sig:
            x = torch.sigmoid(x)
        return x

    def eval_weight(self, x):
        output = self.forward(x)
        weights = output/(1-output)
        return weights

    def apply(self, pos, neg, return_acc=True, requires_grad=True, epoch=0,
              x_samps=None, x_gen=None):
        '''Applies the loss function self.Loss_type onto the discriminator input.
            args:
                self                : [object] Loss
                pos                 : [tensor] discriminator output for true samples
                neg                 : [tensor] discriminator output for fake samples
            kwargs:
                return_acc          : [bool]   return the accuracys of the discriminative model
                requires_grad       : [bool]   attach gradient to the noise, required for gradient penalty
                epoch               : [int] current training epoch
                x_samps, x_gen      : [tensor] needed for gradient penalty
        '''

        #if ("weight_plots" in self.params.get("plots",[]) and
        #        epoch%self.params.get("weight_interval",5) == 0 and
        #        not self.adversarial):
        weights_true = pos/torch.clamp((1-pos), min=1.e-7)
        weights_fake = neg/torch.clamp((1-neg), min=1.e-7)
        #    self.data_store["epoch_weights_true"].append(weights_true.detach().cpu().numpy())
        #    self.data_store["epoch_weights_fake"].append(weights_fake.detach().cpu().numpy())

        # regularization
        regularization = 0

        #if self.params.get("gradient_penalty", 0.0) > 0 and requires_grad:
        #    regularization = self.params.get("gradient_penalty", 0.0) * \
        #    Discriminator_Regularizer(pos, x_samps, neg, x_gen)

        # compute loss
        loss       = self.loss(pos, neg) + regularization
        if return_acc:
            return loss, torch.mean((pos > 0).float()), torch.mean((neg < 0).float())
        else:
            return loss

    def loss(self, pos, neg):
        '''Computes the usual BCE Loss for true and fake data.'''
        return self.BCE(neg, torch.zeros_like(neg)) + \
        self.BCE(pos, torch.ones_like(pos))


