# pylint: disable=invalid-name
""" This classifier should learn the difference between GEANT and CaloFlow (CaloGAN) events.
    If it is unable to tell the difference, the generated samples are realistic.

    Used for
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

    Note: _test files are used for validation, the val_ file is then loaded for final evaluation
"""

######################################   Imports   #################################################
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from data import get_dataloader
import plotting_helper as plthlp


#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# file structures
parser.add_argument('--save_dir', default='./classifier', help='Where to save the trained model')
parser.add_argument('--data_dir', default='/media/claudius/8491-9E93/ML_sources/CaloGAN/classifier',
                    help='Where to find the dataset')
parser.add_argument('--run_number', '-r', default=None, type=str,
                    help='specific run number, to be added to file name')
parser.add_argument('--name', default=None, type=str)

# Calo specific
parser.add_argument('--particle_type', '-p',
                    help='Name of the dataset file w/o extension and "train_" or "test_" prefix')

# NN parameters

# DNN or CNN
parser.add_argument('--mode', default='DNN-low',
                    choices=["DNN-low", "DNN-high", "CNN"],
                    help='must be in ["DNN-low", "DNN-high", "CNN"]')
parser.add_argument('--n_layer', type=int, default=2,
                    help='Number of hidden layers in the classifier.')
parser.add_argument('--n_hidden', type=int, default='512',
                    help='Hidden nodes per layer.')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.,
                    help='dropout probability')
parser.add_argument('--use_logit', action='store_true', help='If data is logit transformed')
parser.add_argument('--threshold', action='store_true', help='If threshold of 1e-2MeV is applied')
parser.add_argument('--normalize', action='store_true',
                    help='If voxels should be normalized per layer')

# training params
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--n_epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=70,
                    help='How often to show loss statistics.')
parser.add_argument('--load', action='store_true', default=False,
                    help='Whether or not load model from --save_dir')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

#######################################   helper functions   #######################################
ALPHA = 1e-6
def logit(x):
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def make_input_low(dataloader, arg):
    """ takes dataloader and returns tensor of
        layer0, layer1, layer2, log10 energy
    """
    layer0 = dataloader['layer_0'].to(arg.dtype)
    layer1 = dataloader['layer_1'].to(arg.dtype)
    layer2 = dataloader['layer_2'].to(arg.dtype)
    energy = torch.log10(dataloader['energy']*10.).to(arg.device).to(arg.dtype)
    E0 = dataloader['layer_0_E'].to(arg.dtype)
    E1 = dataloader['layer_1_E'].to(arg.dtype)
    E2 = dataloader['layer_2_E'].to(arg.dtype)

    if arg.threshold:
        layer0 = torch.where(layer0 < 1e-7, torch.zeros_like(layer0), layer0)
        layer1 = torch.where(layer1 < 1e-7, torch.zeros_like(layer1), layer1)
        layer2 = torch.where(layer2 < 1e-7, torch.zeros_like(layer2), layer2)

    if arg.normalize:
        layer0 = layer0 / (E0.reshape(-1, 1, 1) +1e-16)
        layer1 = layer1 / (E1.reshape(-1, 1, 1) +1e-16)
        layer2 = layer2 / (E2.reshape(-1, 1, 1) +1e-16)

    E0 = (torch.log10(E0.unsqueeze(-1)+1e-8) + 2.).to(arg.device)
    E1 = (torch.log10(E1.unsqueeze(-1)+1e-8) + 2.).to(arg.device)
    E2 = (torch.log10(E2.unsqueeze(-1)+1e-8) + 2.).to(arg.device)

    target = dataloader['label'].to(arg.device)

    layer0 = layer0.view(layer0.shape[0], -1).to(arg.device)
    layer1 = layer1.view(layer1.shape[0], -1).to(arg.device)
    layer2 = layer2.view(layer2.shape[0], -1).to(arg.device)

    if arg.use_logit:
        layer0 = logit_trafo(layer0)/10.
        layer1 = logit_trafo(layer1)/10.
        layer2 = logit_trafo(layer2)/10.

    return torch.cat((layer0, layer1, layer2, energy, E0, E1, E2), 1), target

def make_input_high(dataloader, arg):
    """ returns high-level features, computed from batch of low-level voxels. """
    if arg.threshold:
        cut = 1e-2
    else:
        cut = 0.

    incident_energy = torch.log10(dataloader['energy']*10.).to(arg.device).to(arg.dtype)
    # scale them back to MeV
    layer0 = dataloader['layer_0'].to(arg.dtype) * 1e5
    layer1 = dataloader['layer_1'].to(arg.dtype) * 1e5
    layer2 = dataloader['layer_2'].to(arg.dtype) * 1e5
    layer0 = plthlp.to_np_thres(layer0.view(layer0.shape[0], -1), cut)
    layer1 = plthlp.to_np_thres(layer1.view(layer1.shape[0], -1), cut)
    layer2 = plthlp.to_np_thres(layer2.view(layer2.shape[0], -1), cut)
    # detour to numpy in order to use same functions as plotting script
    full_shower = np.concatenate((layer0, layer1, layer2), 1)
    E_0 = plthlp.energy_sum(layer0)
    E_1 = plthlp.energy_sum(layer1)
    E_2 = plthlp.energy_sum(layer2)
    E_tot = E_0 + E_1 + E_2
    f_0 = E_0 / E_tot
    f_1 = E_1 / E_tot
    f_2 = E_2 / E_tot
    l_d = plthlp.depth_weighted_energy(full_shower)
    s_d = l_d / (E_tot * 1e3)
    sigma_sd = plthlp.depth_weighted_energy_normed_std(full_shower)
    E_1b0, E_2b0, E_3b0, E_4b0, E_5b0 = plthlp.n_brightest_voxel(layer0, [1, 2, 3, 4, 5]).T
    E_1b1, E_2b1, E_3b1, E_4b1, E_5b1 = plthlp.n_brightest_voxel(layer1, [1, 2, 3, 4, 5]).T
    E_1b2, E_2b2, E_3b2, E_4b2, E_5b2 = plthlp.n_brightest_voxel(layer2, [1, 2, 3, 4, 5]).T
    ratio_0 = plthlp.ratio_two_brightest(layer0)
    ratio_1 = plthlp.ratio_two_brightest(layer1)
    ratio_2 = plthlp.ratio_two_brightest(layer2)
    sparsity_0 = plthlp.layer_sparsity(layer0, cut)
    sparsity_1 = plthlp.layer_sparsity(layer1, cut)
    sparsity_2 = plthlp.layer_sparsity(layer2, cut)
    phi_0 = plthlp.center_of_energy(layer0, 0, 'phi')
    phi_1 = plthlp.center_of_energy(layer1, 1, 'phi')
    phi_2 = plthlp.center_of_energy(layer2, 2, 'phi')
    eta_0 = plthlp.center_of_energy(layer0, 0, 'eta')
    eta_1 = plthlp.center_of_energy(layer1, 1, 'eta')
    eta_2 = plthlp.center_of_energy(layer2, 2, 'eta')
    sigma_0 = plthlp.center_of_energy_std(layer0, 0, 'phi')
    sigma_1 = plthlp.center_of_energy_std(layer1, 1, 'phi')
    sigma_2 = plthlp.center_of_energy_std(layer2, 2, 'phi')

    # to be log10-processed:
    ret1 = np.vstack([E_0+1e-8, E_1+1e-8, E_2+1e-8, E_tot,
                      f_0+1e-8, f_1+1e-8, f_2+1e-8, l_d+1e-8,
                      sigma_0+1e-8, sigma_1+1e-8, sigma_2+1e-8]).T
    ret1 = np.log10(ret1)
    # without log10 processing:
    ret2 = np.vstack([s_d, sigma_sd,
                      1e1*E_1b0, 1e1*E_2b0, 1e1*E_3b0, 1e1*E_4b0, 1e1*E_5b0,
                      1e1*E_1b1, 1e1*E_2b1, 1e1*E_3b1, 1e1*E_4b1, 1e1*E_5b1,
                      1e1*E_1b2, 1e1*E_2b2, 1e1*E_3b2, 1e1*E_4b2, 1e1*E_5b2,
                      ratio_0, ratio_1, ratio_2, sparsity_0, sparsity_1, sparsity_2,
                      phi_0/1e2, phi_1/1e2, phi_2/1e2, eta_0/1e2, eta_1/1e2, eta_2/1e2]).T
    ret = torch.from_numpy(np.hstack([ret1, ret2])).to(arg.device).to(arg.dtype)

    ret = torch.cat((ret, incident_energy), 1)
    target = dataloader['label'].to(arg.device)

    return ret, target

def load_classifier(constructed_model, parser_args, filepath=None):
    """ loads a saved model """
    if filepath is None:
        filename = parser_args.mode+'.pt' if parser_args.run_number is None else \
            parser_args.mode + '_' + parser_args.run_number + '_' + parser_args.name + '.pt'
        checkpoint = torch.load(os.path.join(parser_args.save_dir, filename),
                                map_location=args.device)
    else:
        checkpoint = torch.load(filepath,
                                map_location=args.device)
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    return constructed_model

######################################## constructing the NN #######################################

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

class CNN(torch.nn.Module):
    """ CNN for improved classification """
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers_0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=(3, 7), padding=(2, 0)),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=(4, 7), padding=(2, 0)),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 14)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.cnn_layers_2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 1)),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        linear_layers = [
            torch.nn.Linear(256+256+256+4, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1)
        ]
        self.linear_layers = torch.nn.Sequential(*linear_layers)

    def forward(self, x):
        split_elements = [*torch.split(x, [288, 144, 72, 4], dim=-1)]
        layer_0 = split_elements[0].view(x.size(0), 3, 96)
        layer_1 = split_elements[1].view(x.size(0), 12, 12)
        layer_2 = split_elements[2].view(x.size(0), 12, 6)
        layer_0 = self.cnn_layers_0(layer_0.unsqueeze(1)).view(x.size(0), -1)
        layer_1 = self.cnn_layers_1(layer_1.unsqueeze(1)).view(x.size(0), -1)
        layer_2 = self.cnn_layers_2(layer_2.unsqueeze(1)).view(x.size(0), -1)
        all_together = torch.cat((layer_0, layer_1, layer_2, split_elements[3]), 1)
        ret = self.linear_layers(all_together)
        return ret

##################################### train and evaluation functions ###############################
def train_and_evaluate(model, data_train, data_test, optimizer, args):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    args.best_epoch = -1
    log_loss = []
    try:
        for i in range(args.n_epochs):
            t_loss = train(model, data_train, optimizer, i, args)
            log_loss.append(t_loss)
            with torch.no_grad():
                eval_acc, _ = evaluate(model, data_test, i, args)
                #args.test_loss.append(-eval_loss.to('cpu').numpy())
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                args.best_epoch = i+1
                filename = args.mode+'.pt' if args.run_number is None else \
                           args.mode + '_' + args.run_number + '_' + args.name + '.pt'
                torch.save({'model_state_dict':model.state_dict()},
                           os.path.join(args.save_dir, filename))
            if eval_acc == 1.:
                break
        np.save(args.save_dir+'train_loss.npy', np.array(log_loss))
    except KeyboardInterrupt:
        pass

def train(model, train_data, optimizer, epoch, args):
    """ train one step """
    model.train()
    log_loss = []
    for i, data_batch in enumerate(train_data):
        input_vector, target_vector = make_input(data_batch, args)
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        #target_vector[target_vector == 0] += torch.rand_like(target_vector[target_vector==0])*1.0e-1
        #target_vector[target_vector == 1] -= torch.rand_like(target_vector[target_vector==1])*1.0e-1

        loss = criterion(output_vector, target_vector.unsqueeze(1))
        log_loss.append(loss.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (len(train_data)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(train_data), loss.item()))
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), res_pred.cpu()))
    return log_loss

def evaluate(model, test_data, i, args, return_ROC=False, final_eval=False, calibration_data=None):
    """ evaluate on test set """
    model.eval()
    for j, data_batch in enumerate(test_data):
        input_vector, target_vector = make_input(data_batch, args)
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.to(args.dtype)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)

    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    JSD = - BCE + np.log(2.)
    print("BCE loss of test set is {}, JSD of the two dists is {}".format(BCE, JSD/np.log(2.)))
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred)
        calibrator = calibrate_classifier(model, calibration_data, args)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        print("rescaled AUC of dataset is", eval_auc)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        print("rescaled calibration curve:", prob_true, prob_pred)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
        JSD = - BCE.cpu().numpy() + np.log(2.)
        otp_str = "rescaled BCE loss of test set is {}, rescaled JSD of the two dists is {}"
        print(otp_str.format(BCE, JSD/np.log(2.)))
        #write to file if run_number exists
        if args.run_number is not None:
            results = np.array([[eval_acc, eval_auc, JSD/np.log(2.), args.best_epoch]])
            filename = 'summary_'+('loaded_' if args.load else '')+args.mode+'_'+args.name+'.npy'
            if args.run_number == '1':
                np.save(os.path.join(args.save_dir, filename), results)
            else:
                prev_res = np.load(os.path.join(args.save_dir, filename),
                                   allow_pickle=True)
                new_res = np.concatenate([prev_res, results])
                np.save(os.path.join(args.save_dir, filename), new_res)
    if not return_ROC:
        return eval_acc, JSD #, eval_loss_tot.mean(0)
    else:
        return roc_curve(result_true, result_pred)

def calibrate_classifier(model, calibration_data, args):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        input_vector, target_vector = make_input(data_batch, args)
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(args.dtype)
        if j == 0:
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

####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    if args.mode == 'CNN':
        args.dtype = torch.float
    else:
        args.dtype = torch.float64
    torch.set_default_dtype(args.dtype)

    # set up device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # low: voxel+E_i+E_inc; high: as in figures of paper; CNN: only low possible
    input_dim = {'DNN-low': 508, 'DNN-high': 41, 'CNN': None}[args.mode]
    DNN_kwargs = {'num_layer':args.n_layer,
                  'num_hidden':args.n_hidden,
                  'input_dim':input_dim,
                  'dropout_probability':args.dropout_probability}
    if 'DNN' in args.mode:
        model = DNN(**DNN_kwargs)
    else:
        model = CNN()
    model.to(args.device)
    print(model)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("{} has {} parameters".format(args.mode, int(total_parameters)))

    if 'high' in args.mode:
        make_input = make_input_high
    else:
        make_input = make_input_low

    # test used for model selection!
    data_train, data_test = get_dataloader(args.particle_type,
                                           args.data_dir,
                                           full=False,
                                           apply_logit=False,
                                           device=args.device,
                                           batch_size=args.batch_size,
                                           with_noise=False,
                                           normed=False,
                                           normed_layer=False,
                                           return_label=True)
    # val used for final score
    data_val = get_dataloader('val_'+args.particle_type,
                              args.data_dir,
                              full=True,
                              apply_logit=False,
                              device=args.device,
                              batch_size=args.batch_size,
                              with_noise=False,
                              normed=False,
                              normed_layer=False,
                              return_label=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.test_loss = []

    if args.load:
        model = load_classifier(model, args)
        args.best_epoch = -1
    else:
        train_and_evaluate(model, data_train, data_test, optimizer, args)

    model = load_classifier(model, args)
    with torch.no_grad():
        print("Now looking at independent dataset:")
        eval_logprob = evaluate(model, data_val, 0, args, final_eval=True,
                                calibration_data=data_test)
