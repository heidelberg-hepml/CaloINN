import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas as pd
from matplotlib import cm
# from matplotlib.transforms import Bbox

import data_util
# from calc_obs import *
import math
import torch

from evaluate_plotting_helper import *

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.default'] = 'rm'
plt.rcParams['text.usetex'] = True

labelfont = FontProperties()
labelfont.set_family('serif')
labelfont.set_name('Times New Roman')
labelfont.set_size(20)

axislabelfont = FontProperties()
axislabelfont.set_family('serif')
axislabelfont.set_name('Times New Roman')
axislabelfont.set_size(20)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_name('Times New Roman')
tickfont.set_size(20)



def plot_average_table(data, save_file):
    
    number_of_runs = len(data)
    row_indices = [f"Run {i+1}" for i in range(number_of_runs)]

    row_indices.append("mean")
    row_indices.append("std")

    averaged = np.array([np.mean(data, axis=0), np.std(data, axis=0)])
    data = np.concatenate([data, averaged])

    df = pd.DataFrame(data, columns=["Accuracy", "AUC", "JSD", "best epoch"], index=row_indices)

    number_of_cols = len(df.columns)
    number_of_rows = len(list(df.iterrows()))
    row_labels = np.array(list(df.iterrows()), dtype=object)[:,0]
    fig, ax = plt.subplots(figsize=((number_of_rows+1), (number_of_cols)))

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values.round(4), colLabels=df.columns, rowLabels=row_labels,
            loc='center', cellLoc="left",colWidths = [0.1]*len(df.columns),
            colLoc="center", rowLoc="center")
    fig.tight_layout()

    plt.savefig(save_file)
    plt.close()

def plot_loss(
        file_name,
        loss_train,
        loss_test,
        skip_epochs=True):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    c = len(loss_test)/len(loss_train)
    ax.plot(c*np.arange(1,len(loss_train)+1), loss_train, color='blue', label='train loss')
    ax.plot(np.arange(1,len(loss_test)+1), loss_test, color='red', label='test loss')
    ax.legend(loc='upper right', prop=labelfont)

    ax.set_xlim([0,len(loss_test)])
    # nested np.mins needed for the case of different length
    # print(len(loss_test))
    if len(loss_test) <= 10 or (not skip_epochs):
        y_min = np.min(np.min(np.array([loss_train, loss_test], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train, loss_test], dtype=object)))
    elif len(loss_test) <= 20:
        train_idx = 10 * len(loss_train) // len(loss_test)
        y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
    # elif len(loss_test) <= 20:
    else:
        train_idx = 20 * len(loss_train) // len(loss_test)
        y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
        y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
        
    # print(y_min, y_max)
    if y_min > 0:
        if y_max > 0:
            ax.set_ylim([y_min*0.9, y_max*1.1])
        else:
            ax.set_ylim([y_min*0.9, y_max*0.9])
    else:
        if y_max > 0:
            ax.set_ylim([y_min*1.1, y_max*1.1])
        else:
            ax.set_ylim([y_min*1.1, y_max*0.9])
            
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('loss', fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()

def plot_lr(
        file_name,
        learning_rate,
        batches_per_epoch=1):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    ax.plot(np.arange(1,len(learning_rate)+1)/batches_per_epoch, learning_rate, color='red', label='learning rate')

    ax.set_xlim([0,len(learning_rate)/batches_per_epoch])
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('learning rate', fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
    
def plot_grad(
        file_name,
        gradients,
        batches_per_epoch=1):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    ax.plot(np.arange(1,len(gradients)+1)/batches_per_epoch, gradients, color='red', label='gradient')

    ax.set_xlim([0,len(gradients)/batches_per_epoch])
    ax.set_xlabel('epoch', fontproperties=axislabelfont)
    ax.set_ylabel('gradient', fontproperties=axislabelfont)
    ax.set_yscale("log")

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
    
def plot_logsig(
        file_name,
        logsigs):
    
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    colors = ["red", "blue", "green", "orange"]
    labels = ["max", "min", "mean", "median"]
    for logsig, label, color in zip(logsigs, labels, colors):
        
        ax.plot(logsig, label=label, color=color)

        ax.set_xlim([0,len(logsig)])
        ax.set_xlabel('epoch', fontproperties=axislabelfont)
        ax.set_ylabel('$log(\\sigma^2)$', fontproperties=axislabelfont)


    ax.legend()
    fig.tight_layout()
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()
  
def plot_all_hist(x_true, c_true, x_fake, c_fake, params, layer_boundaries, plot_dir, threshold=1.e-10):
    
    os.makedirs(plot_dir, exist_ok=False)
    
    def get_args_for_plotting(params, plot_dir):
        """Returns a args element for the plotting"""
        # TODO: Needs update for dataset 2 & 3
        
        parser_replacement = {"dataset" : params["particle_type"] + "s",
                            "output_dir" : plot_dir,
                            "mode" : "all",
                            "x_scale": "log",
                            "min_energy": 10}
        
        
        args = argparse.Namespace(**parser_replacement)
        
        args.min_energy = {'photons': 10, 'pions': 10,
                        '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]
        
        
        return args

    def plot(hlf_true, hlf_fake, args):
        plot_Etot_Einc(hlf_fake, hlf_true, args)
        plot_E_layers(hlf_fake, hlf_true, args)
        plot_ECEtas(hlf_fake, hlf_true, args)
        plot_ECPhis(hlf_fake, hlf_true, args)
        plot_ECWidthEtas(hlf_fake, hlf_true, args)
        plot_ECWidthPhis(hlf_fake, hlf_true, args)
        if args.dataset[0] == '1':
            plot_Etot_Einc_discrete(hlf_fake, hlf_true, args)
            
    args = get_args_for_plotting(params, plot_dir=plot_dir)
    hlf_true = data_util.get_hlf(x_true, c_true, params["particle_type"], layer_boundaries, threshold=1.e-10)
    hlf_fake = data_util.get_hlf(x_fake, c_fake, params["particle_type"], layer_boundaries, threshold=1.e-10)
    
    plot(hlf_true, hlf_fake, args)
          
def plot_latent(samples, results_dir, epoch=None):
    if epoch is not None:
        plot_dir = os.path.join(results_dir, 'latent', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'latent')
    os.makedirs(plot_dir, exist_ok=True)
    
    max_dims = samples.shape[1]
    
    # previously:
    # latent_dims = [1, 150, 300, 400, 500, 504, 505, 506]
    np.linspace(0, max_dims-4, 5)
    
    # Cover the space equally and look at the extra dims dimensions
    latent_dims = list(np.linspace(0, max_dims-4, 5).astype(int)) + [max_dims-3, max_dims-2, max_dims-1]
        
    
    for idx in latent_dims:
        min_v = -3
        max_v = 3
        bins = np.linspace(min_v, max_v, 51)

        fig,axs = plt.subplots(1,1,figsize=(6,4))

        axs.hist(samples[:,idx], bins, color='red', histtype='step', density=True)

        axs.set_xlim(min_v, max_v)

        z_ = np.linspace(min_v, max_v, 501)
        p = 1/np.sqrt(2*np.pi)*np.exp(-z_**2/2)
        axs.plot(z_, p, color='black')

        plt.xticks(fontproperties=tickfont)
        plt.yticks(fontproperties=tickfont)

        axs.set_xlabel(f'\\(z_{{{idx+1}}}\\)', fontproperties=axislabelfont)
        axs.set_ylabel('normalized distribution', fontproperties=axislabelfont)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'latent_{idx:03d}.pdf'), bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='Where to find the results and save the plots')
    parser.add_argument('--reference_file', help='Where to find the reference data')
    parser.add_argument('--include_coro', action='store_true', help='Also plot the pixel to pixel correlation (Computationally expensive)')
    parser.add_argument('--layer', default=None, type=int, help='Which layer to plot')
    args = parser.parse_args()

    # plot_all_hist(args.results_dir, args.reference_file, args.include_coro, args.layer)

if __name__=='__main__':
    main()
