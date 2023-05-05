import argparse
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import data_util
from model import CINN
from documenter import Documenter
import yaml

parser = argparse.ArgumentParser()

# file structures
parser.add_argument('--param_file', default='.params/example.yaml', help='param file .yaml')
parser.add_argument('--save_dir', default='./classifier', help='Where to save the trained model')
parser.add_argument('--data_dir', default='/media/claudius/8491-9E93/ML_sources/CaloGAN/classifier',
                    help='Where to find the dataset')

# Calo specific
parser.add_argument('--particle_type', '-p',
                    help='Name of the dataset file w/o extension and "train_" or "test_" prefix')

parser.add_argument('--n_layer', type=int, default=2,
                    help='Number of hidden layers in the classifier.')
parser.add_argument('--n_hidden', type=int, default='512',
                    help='Hidden nodes per layer.')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability')

# training params
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=70,
                    help='How often to show loss statistics.')
parser.add_argument('--load', action='store_true', default=False,
                    help='Whether or not load model from --save_dir')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

class DNN_model(nn.Module):
    """ NN for vanilla classifier """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.,
                 is_classifier=True):
        super(DNN_model, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.ReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        all_layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class DNN_trainer:
    'dnn classifier test'
    def __init__(self, params, device, log):
        """
            params: dict with training parameters
            device: device cuda/cpu
        """
        self.params = params
        self.device = device
        self.log = log
        train_loader, test_loader = data_util.get_loaders(
				params.get('train_data_path'),
				params.get('test_data_path'),
				params.get('batch_size'),
				params.get('train_split', 0.8),
				device,
				params.get("width_noise", 1e-7),
				params.get("use_extra_dim", False),
				params.get("use_extra_dims", False),
				params.get("mask", 0),
				params.get("calo_layer", None),
                params.get("is_classifier", False)
			)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def make_input(self, loader, ener):
        """ takes dataloader and returns tensor of
            layer0, layer1, layer2, log10 energy
        """
        layer0 = loader[:, :288].to(self.device)
        layer1 = loader[:, 288:432].to(self.device)
        layer2 = loader[:, 432:-1].to(self.device)
        energy = torch.log(ener).to(self.device)

        en0 = torch.sum(layer0, dim=1, keepdims=True).to(self.device)
        en1 = torch.sum(layer1, dim=1, keepdims=True).to(self.device)
        en2 = torch.sum(layer2, dim=1, keepdims=True).to(self.device)

        th_noise = self.params.get("width_noise", None)
        threshold = self.params.get("remove_noise", None)
        if threshold:
            layer0 = torch.where(layer0 < th_noise, torch.zeros_like(layer0), layer0)
            layer1 = torch.where(layer1 < th_noise, torch.zeros_like(layer1), layer1)
            layer2 = torch.where(layer2 < th_noise, torch.zeros_like(layer2), layer2)

        target = loader[:, -1].to(self.device)
        return torch.cat((layer0, layer1, layer2, energy, en0, en1, en2), 1), target

    def train(self):
        n_layer = self.params.get("n_layer", 3)
        n_hidden = self.params.get("n_hidden", 512)
        dout = self.params.get("dropout_probability", 0.0)
        lr = self.params.get("lr", 1e-4)
        n_epochs = self.params.get("n_epochs", 60)

        model = DNN_model(n_layer, n_hidden, self.make_input(self.train_loader.data[:2], self.train_loader.cond[:2])[0].shape[1], dout)
        model.to(self.device)
        model.train()
        crit = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr)

        best_eval_acc = 0.0
        best_eval_loss = 100.
        best_eval_AUC = 0.0
        for epoch in range(n_epochs):
            model.train()
            for i, batch in enumerate(self.train_loader):
                inputs, labels = self.make_input(batch[0], batch[1])
                inputs = torch.nan_to_num(inputs, nan=0, neginf=0, posinf=0)
                pred = model(inputs)
                labels = torch.round(labels)
                loss = crit(pred, labels.unsqueeze(1))

                opt.zero_grad()
                loss.backward()
                opt.step()

                if i % (len(self.train_loader.data)//2) == 0:
                    print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, n_epochs, i, len(self.train_loader.data), loss.item()))
                # PREDICTIONS
                #pred = np.round(output_vector.detach().cpu())
                #target = np.round(target_vector.detach().cpu())
                #res_pred.extend(pred.tolist())
                #res_true.extend(target.tolist())
                pred_r = torch.round(pred.detach())
                true = torch.round(labels.detach())
                if i == 0:
                    res_true = true
                    res_pred = pred_r
                else:
                    res_true = torch.cat((res_true, true), 0)
                    res_pred = torch.cat((res_pred, pred_r), 0)

            print("Accuracy on training set is",
                accuracy_score(res_true.cpu(), res_pred.cpu()))
       
            model.eval()
            test_preds = []
            test_labs = []
            test_loss = []
            test_l = nn.BCEWithLogitsLoss()
            with torch.no_grad():
                for i, batch in enumerate(self.test_loader):
                    inputs, labels = self.make_input(batch[0], batch[1])
                    inputs = torch.nan_to_num(inputs, nan=0, neginf=0, posinf=0)
                    pred = model(inputs)
                    labels = torch.round(labels)
                    
                    test_preds.append(pred.detach().cpu().numpy())
                    test_labs.append(labels.detach().cpu().numpy())
                
            test_labs = np.array(test_labs).flatten()
            test_preds = np.array(test_preds).flatten()
            test_loss = test_l( torch.tensor(test_preds), torch.tensor(test_labs)).numpy().mean()

            pred_r = np.round(test_preds)
            lab_r = np.round(test_labs)
            
            log_1 = np.log(test_preds+1.e-7)[test_labs==1]
            log_0 = np.log(1.-test_preds+1.e-7)[test_labs==0]

            BCE = np.mean(log_1) + np.mean(log_0)
            JSD = 0.5*BCE + np.log(2.)
            eval_acc = accuracy_score(lab_r, pred_r)
            eval_loss = test_loss
            eval_AUC = roc_auc_score(test_labs, test_preds)
            print("Accuracy on test set is", eval_acc)
            print("AUC on test set is", eval_AUC)
            print("BCE loss of test set is {}, JSD of the two dists is {}".format(-BCE, JSD/np.log(2.)))
            print("Test loss with Pytorch ", eval_loss)
            print(" ")
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save({'model_state_dict':model.state_dict(),
                            'epoch': epoch},
                           self.log.get_file(f"class_best_loss.pt"))

            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                torch.save({'model_state_dict':model.state_dict(),
                            'epoch': epoch},
                           self.log.get_file(f"class_best_acc.pt"))


            if eval_AUC > best_eval_AUC:
                best_eval_AUC = eval_AUC
                torch.save({'model_state_dict':model.state_dict(),
                            'epoch': epoch},
                           self.log.get_file(f"class_best_AUC.pt"))


if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    # file structures
    parser.add_argument('--param_file', default='.params/example.yaml', help='param file .yaml')
    parser.add_argument('--save_dir', default='./classifier', help='Where to save the trained model')
    parser.add_argument('--data_dir', default='/media/claudius/8491-9E93/ML_sources/CaloGAN/classifier',
                        help='Where to find the dataset')

    # Calo specific
    parser.add_argument('--particle_type', '-p',
                        help='Name of the dataset file w/o extension and "train_" or "test_" prefix')

    # CUDA parameters
    parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')

    args = parser.parse_args()
    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    device = 'cuda:0' if not args.no_cuda else 'cpu'
    doc = Documenter(params['run_name'])
    trainer = DNN_trainer(params, device, doc)
    trainer.train()
