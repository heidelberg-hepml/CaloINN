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

    def forward(self, x):
        x = self.layers(x)
        return x

    def eval_weight(self, x):
        output = self.forward(x)
        weights = output/(1-output)
        return weights

