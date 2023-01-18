import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas as pd
from matplotlib import cm
# from matplotlib.transforms import Bbox

import data_util
from calc_obs import *
import math
import torch

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

def plot_hist(
        file_name,
        data,
        reference,
        p_ref='eplus',
        axis_label=None,
        xscale='linear',
        yscale='log',
        vmin=None,
        vmax=None,
        n_bins=100,
        ymin=None,
        ymax=None,
        ax=None,
        panel_ax=None,
        panel_scale="log",
        density=True):
    data = data[np.isfinite(data)]
    reference = reference[np.isfinite(reference)]

    if vmin is None:
        vmin = min(np.min(data), np.min(reference))
    if vmax is None:
        vmax = max(np.max(data), np.max(reference))
    if xscale=='log':
        if vmin==0:
            vmin = min(np.min(data[data>1e-7]), np.min(reference[reference>1e-7]))
        if isinstance(n_bins, int):
            bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins)
        else:
            bins = n_bins
    else:
        if isinstance(n_bins, int):
            bins = np.linspace(vmin, vmax, n_bins)
        else:
            bins = n_bins
    
    colors = cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    if p_ref == 'eplus':
        color = colors[0]
    elif p_ref == 'gamma':
        color = colors[1]
    elif p_ref == 'piplus':
        color = colors[2]
    else:
        color = 'blue'
        
    create_fig = False
    if ax is None:
        create_fig = True
        fig, ax = plt.subplots(1,1,figsize=(6,6))

    ns_0, bins_0, patches_0 = ax.hist(data, bins=bins, histtype='step', linewidth=2,
        alpha=1, color=color, density=density, label='CaloINN')

    ns_1, bins_1, patches_1 = ax.hist(reference, bins=bins, histtype='stepfilled',
            alpha=0.5, color=color, density=density, label='GEANT')

    if panel_ax is not None:
        assert len(bins_0) == len(bins_1)
        assert (bins_0 - bins_1 < 1.e-7).all()
        
        # prevent divisions by 0! Set these bars to 0
        mask = ns_1 == 0
        ns_1[mask] = 1
        panel_data = ns_0/ns_1
        
        panel_data[mask] = float("nan")
        
        widths = 1.2*(bins_1[1:] - bins_1[:-1])
        
        panel_ax.bar(bins_0[:-1], ns_0/ns_1, label='CaloINN/GEANT', width=widths)
        
        panel_ax.plot([vmin, vmax],[1,1], color="red", ls="--", marker=None)

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if panel_ax is not None:
        panel_ax.set_yscale(panel_scale)
        panel_ax.set_xscale(xscale)

    ax.set_xlim([vmin,vmax])
    if panel_ax is not None:
        panel_ax.set_xlim([vmin,vmax])
        # panel_ax.set_ylim([0.9, 1.1])
        
    if ymin is not None or ymax is not None:
        ax.set_ylim((ymin, ymax))

    if panel_ax is not None:
        panel_ax.legend()

    if axis_label:
        if panel_ax is None:
            ax.set_xlabel(axis_label, fontproperties=axislabelfont)
        else:
            panel_ax.set_xlabel(axis_label, fontproperties=axislabelfont)
            
    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    if create_fig:
        fig.tight_layout()
        fig.savefig(file_name, bbox_inches='tight')
        
    if panel_ax is None:
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
    
def plot_all_hist(results_dir, reference_file, include_coro=False, 
                  calo_layer=None, epoch=None, summary_plot=True, single_plots=False,
                  p_ref="e_plus", data=None):
    
    # Load the sampled data if no data is passed:
    if data is None:
        data_file = os.path.join(results_dir, 'samples.hdf5')
        data = data_util.load_data(data_file)
        
    # Load the reference data
    reference = data_util.load_data(reference_file)
    
    # Select the output dir
    if epoch:
        plot_dir = os.path.join(results_dir, 'plots', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'plots/final')
        
    os.makedirs(plot_dir, exist_ok=True)
  
    # Define some plot paramters
    if calo_layer is None:
        plots = [
                (calc_e_ratio, 'e_ratio.pdf', {}, {'axis_label': r'\(E_{tot}/E_{part}\)', 'p_ref': p_ref}),
            (calc_e_ratio, 'e_ratio_log.pdf', {}, {'axis_label': r'\(E_{tot}/E_{part}\)', 'xscale': 'log', 'p_ref': p_ref}),
            (calc_e_detector, 'e_detector.pdf', {}, 
                {'axis_label': r'\(E_{tot}\) (GeV)', 'ymin': 5e-6, 'ymax': 2e-1, 'yscale': 'log', 'vmin': -5, 'vmax': 105, 'n_bins': np.linspace(0, 120, 50), 'p_ref': p_ref}),

            (calc_layer_diff, 'eta_diff_0_1.pdf', {'layer2': 1, 'dir': 'eta'},
                {'axis_label': r'\(\left<\eta_1\right>-\left<\eta_0\right>\)', 'p_ref': p_ref}),
            (calc_layer_diff, 'eta_diff_0_2.pdf', {'layer2': 2, 'dir': 'eta'},
                {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_0\right>\)', 'p_ref': p_ref}),
            (calc_layer_diff, 'eta_diff_1_2.pdf', {'layer1': 1, 'layer2': 2, 'dir': 'eta'},
                {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_1\right>\)', 'p_ref': p_ref}),

            (calc_layer_diff, 'phi_diff_0_1.pdf', {'layer2': 1, 'dir': 'phi'},
                {'axis_label': r'\(\left<\phi_1\right>-\left<\phi_0\right>\)', 'p_ref': p_ref}),
            (calc_layer_diff, 'phi_diff_0_2.pdf', {'layer2': 2, 'dir': 'phi'},
                {'axis_label': r'\(\left<\phi_2\right>-\left<\phi_0\right>\)', 'p_ref': p_ref}),
            (calc_layer_diff, 'phi_diff_1_2.pdf', {'layer1': 1, 'layer2': 2, 'dir': 'phi'},
                {'axis_label': r'\(\left<\phi_2\right>-\left<\phi_1\right>\)', 'p_ref': p_ref}),
            (calc_depth_weighted_total_energy, 'depth_weighted_tot_e.pdf', {}, 
                {'axis_label': r'lateral depth \(l_d\)', 'yscale': 'log', 'xscale': 'log', 'vmin': 1e1, 'vmax': 1e5, 'ymin': 1e-8, 'ymax': 3e-2, 'n_bins': np.logspace(1, 5, 100), 'p_ref': p_ref}),
            (calc_depth_weighted_total_energy_normed, 'depth_weighted_tot_e_normd.pdf', {},
                {'axis_label': r'shower depth \(s_d\)', 'yscale': 'linear', 'ymin': 0, 'ymax': 7, 'vmin': 0.35, 'vmax': 2.05, 'n_bins': np.linspace(0.4, 2, 100), 'p_ref': p_ref}),
            (calc_depth_weighted_total_energy_std, 'depth_weighted_tot_e_normd_std.pdf', {}, 
                {'axis_label': r'shower dept width \(\sigma_{s_d}\)', 'yscale': 'linear', 'ymin': 0, 'ymax': 7, 'vmin': -0.03, 'vmax': 0.93, 'n_bins': np.linspace(0., 0.9, 100), 'p_ref': p_ref}),
        ]
    else:
        plots = []

    for layer in ([0,1,2] if calo_layer is None else [calo_layer]):
        bins_e = [np.logspace(-2, 2, 100), np.logspace(-1, 3, 100), np.logspace(-2, 2, 100)]
        bins_er = [np.logspace(-4, 0, 100), np.logspace(-1, 0, 100), np.logspace(-4, 1, 100)]
        if calo_layer is None:
            plots.append( (calc_e_layer_normd, f'e_normd_layer_{layer}.pdf', {'layer': layer},
                {'axis_label': f'\\(E_{layer}/E_{{tot}}\\)', 'p_ref': p_ref}) )
            plots.append( (calc_e_layer_normd, f'e_normd_layer_{layer}_log.pdf', {'layer': layer},
                {'axis_label': f'\\(E_{layer}/E_{{tot}}\\)', 'xscale': 'log', 'yscale': 'log', 'vmin': (9e-5, 9e-2, 9e-5)[layer], 'vmax': (1.1e0, 1.1e0, 1.1e1)[layer], 'n_bins': bins_er[layer], 'p_ref': p_ref}) )

        plots.append( (calc_e_layer, f'e_layer_{layer}.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}\\) (GeV)', 'p_ref': p_ref}) )
        plots.append( (calc_e_layer, f'e_layer_{layer}_log.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}\\) (GeV)', 'xscale': 'log', 'yscale': 'log', 'n_bins': bins_e[layer], 'vmax': (40, 140, 100)[layer], 'vmin': (6e-3, 6e-2, 6e-3)[layer], 'ymin': 1e-6, 'ymax': 5e1, 'p_ref': p_ref}) )

        plots.append( (calc_sparsity, f'sparsity_{layer}.pdf', {'layer': layer},
            {'axis_label': f'sparsity layer {layer}', 'xscale': 'linear', 'yscale': 'linear', 'n_bins': np.linspace(0, 1, 20), 'vmin': -0.05, 'vmax': 1.05, 'p_ref': p_ref}) )
        plots.append( (calc_layer_brightest_ratio, f'e_ratio_{layer}.pdf', {'layer': layer}, 
            {'axis_label': f'ratio \\(E_{layer}\\)', 'vmin': -0.05, 'vmax': 1.05, 'yscale': 'linear', 'n_bins': np.linspace(0, 1, 100), 'p_ref': p_ref}) )

        for dir in ['eta', 'phi']:
            plots.append( (calc_centroid_mean, f'{dir}_{layer}.pdf', {'layer': layer, 'dir': dir},
                {'axis_label': f'\\(\\left<\\{dir}_{layer}\\right>\\)', 'yscale': 'log', 'vmin': -130, 'vmax': 130, 'n_bins': np.linspace(-125, 125, 50), 'p_ref': p_ref}) )
            plots.append( (calc_centroid_std, f'{dir}_{layer}_std.pdf', {'layer': layer, 'dir': dir},
                {'axis_label': f'std \\(\\{dir}_{layer}\\)', 'vmin': 7e-1, 'vmax': 2e2, 'xscale': 'log', 'yscale': 'log', 'ymin': None, 'ymax': 1e0, 'n_bins': (np.logspace(0, 3, 100), np.logspace(0, 2, 100), np.logspace(0, 3, 100))[layer], 'p_ref': p_ref}) )

        for N in range(1,6):
            plots.append( (calc_brightest_voxel, f'{N}_brightest_voxel_layer_{layer}.pdf', {'layer': layer, 'N': N},
                {'axis_label': f'{N}. brightest voxel in layer {layer}', 'yscale': 'linear', 'vmin': -0.05/N, 'vmax': 1/N+0.05/N, 'n_bins': np.linspace(0., 1/N, 100), 'p_ref': p_ref}) )

        plots.append( (calc_spectrum, f'spectrum_{layer}.pdf', {'layer': layer},
            {'axis_label': f'voxel energy (GeV)', 'xscale': 'log', 'yscale': 'log', 'p_ref': p_ref}) )

        if include_coro:
            plots.append( (calc_coro, f'coro02_{layer}.pdf', {'layer': layer},
                {'axis_label': f'\\(C_{{0.2}}\\) layer {layer}', 'xscale': 'linear', 'yscale': 'log', 'p_ref': p_ref}) )
 
    # Plot every histrogramn in its own file
    if single_plots:
        for function, name, args1, args2 in plots:
            data_coppy = {k: np.copy(v) for k, v in data.items()}
            reference_coppy = {k: np.copy(v) for k, v in reference.items()}
            plot_hist(
                file_name=os.path.join(plot_dir, name),
                data=function(data_coppy, **args1),
                reference=function(reference_coppy, **args1),
                **args2
            )

    # Plot all the histogramms in one file
    if summary_plot:
        number_of_plots = len(plots)
        rows = number_of_plots // 6
        if number_of_plots%6 != 0:
            rows += 1
        heights = [1, 0.3, 0.3]*rows

        fig, axs = plt.subplots(rows*3,6, dpi=500, figsize=(6*7,6*np.sum(heights)), gridspec_kw={'height_ratios': heights})

        data_coppy = {k: np.copy(v) for k, v in data.items()}
        reference_coppy = {k: np.copy(v) for k, v in reference.items()}

        iteration = 0
        for i in range(rows*3):
            
            if i%3 == 1:
                iteration -= 6
                
            for j in range(6):
                
                if i % 3 == 2:
                    # Add one (small) invisible plot as whitespace
                    axs[i,j].set_visible(False)
                    continue
                
                elif iteration >= number_of_plots:
                        # Plots are empty remove them
                        axs[i,j].set_visible(False)
                        iteration += 1
                        continue
                
                
                # Select the correct plot input for this axis
                function, name, args1, args2 = plots[iteration]
                
                
                if i % 3 == 0:
                    # plot the main data
                    plot_hist(
                            file_name=None,
                            data=function(data_coppy, **args1),
                            reference=function(reference_coppy, **args1),
                            ax=axs[i,j],
                            panel_ax=axs[i+1,j],
                            **args2)
                    
                    # Hide the (shared) x-axis
                    axs[i,j].xaxis.set_visible(False)
                    
                    # Hide the first tick label
                    plt.setp(axs[i,j].get_yticklabels()[0], visible=False)  
                    iteration += 1

                if i % 3 == 1:
                    plt.setp(axs[i,j].get_yticklabels()[-1], visible=False)
                    iteration += 1             

        fig.subplots_adjust(hspace=0)
        # fig.savefig(os.path.join(os.path.join(plot_dir,"../"), "final.pdf"), bbox_inches='tight', dpi=500)
        fig.savefig(plot_dir+"/summary.pdf", bbox_inches='tight', dpi=500)
        # Dont use tight_layout!
        plt.close()
        
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

def plot_lin_log_voxels(trainer, epoch, num_samples=100000, n_bins=100, max_width = 6):
    """Plots all the histogramms of the voxels of the passed trainer in linear and logarithmic scale."""
    
    trainer.model.eval()
    
    colors = cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    
    p_ref = trainer.params.get("particle_type", "piplus")
    
    if p_ref == 'eplus':
        color = colors[0]
    elif p_ref == 'gamma':
        color = colors[1]
    elif p_ref == 'piplus':
        color = colors[2]
    else:
        color = 'blue'
    
    generated = trainer.generate(num_samples, return_data=True, save_data=False, postprocessing=False)
    
    num_voxels = trainer.num_dim
    
    rows = math.ceil(num_voxels / max_width)
    
    if rows == 1:
        width = num_voxels
    else: width = max_width
    
    fig, axs = plt.subplots(2*rows,width, figsize=(6*width, 6*2*rows))
    for i, ax in enumerate(axs.flatten()):
        if i < num_voxels:
            data_gen = np.copy(generated[:,i][np.argwhere(np.isfinite(generated[:,i]))])
            data_tst = np.copy(trainer.test_loader.data[:,i].cpu())
            v_min = min([np.nanmin(data_gen), np.nanmin(data_tst)])
            v_max = max([np.nanmax(data_gen), np.nanmax(data_tst)])
            bins = np.linspace(v_min, v_max, n_bins)
            ax.hist(data_gen, bins=bins, density=True, color=color,histtype="step", linewidth=2)
            ax.hist(data_tst, bins=bins, color=color, density=True, alpha=0.5)
            ax.set_title(f"Distribution of {trainer.voxels_list[i]}")
        
        else:
            data_gen = np.copy(generated[:,i-num_voxels][np.argwhere(np.isfinite(generated[:,i-num_voxels]))])
            data_tst = np.copy(trainer.test_loader.data[:,i-num_voxels].cpu())
            data_gen = data_gen[data_gen>1.e-7]
            data_tst = data_tst[data_tst>1.e-7]
            v_min = min([np.nanmin(data_gen), np.nanmin(data_tst)])
            v_max = max([np.nanmax(data_gen), np.nanmax(data_tst)])
            bins = np.logspace(np.log10(v_min), np.log10(v_max), n_bins)
            ax.hist(data_gen, bins=bins, density=True, color=color,histtype="step", linewidth=2)
            ax.hist(data_tst, bins=bins, color=color, density=True, alpha=0.5)
            ax.set_title(f"Distribution of {trainer.voxels_list[i-num_voxels]} (loglog)")
            ax.loglog()

    fig.savefig(trainer.doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "summary.pdf")))
    plt.close()

def plot_overview(save_name, train_loss, train_inn_loss, test_loss, test_inn_loss, 
                           learning_rate, close_to_prior, logsigs, logsig2_prior, batches_per_epoch):
      
    # Logsigma development
    fig, axs = plt.subplots(5, sharex=True, dpi=300, figsize=(12,12), gridspec_kw={'height_ratios': [5,1.5,1,1.5,1.5]})
    colors = ["red", "blue", "green", "orange"]
    labels = ["max", "min", "mean", "median"]

    # Plot the logsigma development
    for logsig, label, color in zip(logsigs, labels, colors):
        axs[0].plot(np.arange(1,len(logsig)+1), logsig, label=label, color=color)
        axs[0].set_ylabel('$log(\\sigma^2)$', fontproperties=axislabelfont)
    axs[0].legend(prop=labelfont)

    axs[1].plot(np.arange(1,len(close_to_prior)+1), close_to_prior, label=f"\# close to the prior of {logsig2_prior:0.2f}", color="red")
    axs[1].set_ylabel('Count', fontproperties=axislabelfont)
    axs[1].legend(prop=labelfont)


    # Plot the LR panel
    axs[2].plot(np.arange(1,len(learning_rate)+1)/batches_per_epoch, learning_rate, color='red', label='learning rate')
    axs[2].set_ylabel('learning rate', fontproperties=axislabelfont)
    axs[2].legend(prop=labelfont)

    # Plot the loss panel
    losses = [[train_loss, test_loss], [train_inn_loss, test_inn_loss]]
    labels_train = ["Total train loss", "Inn train loss"]
    labels_test = ["Total test loss", "Inn test loss"]
    for i, (loss_train, loss_test) in enumerate(losses):
        c = len(loss_test)/len(loss_train)
        axs[3+i].plot(c*np.arange(1,len(loss_train)+1), loss_train, color='blue', label=labels_train[i])
        axs[3+i].plot(np.arange(1,len(loss_test)+1), loss_test, color='red', label=labels_test[i])
        axs[3+i].legend(loc='upper right', prop=labelfont)
        axs[3+i].set_xlim([0,len(loss_test)])
        if len(loss_test) <= 10:
            y_min = np.min(np.min(np.array([loss_train, loss_test], dtype=object)))
            y_max = np.max(np.max(np.array([loss_train, loss_test], dtype=object)))
        elif len(loss_test) <= 20:
            train_idx = 10 * len(loss_train) // len(loss_test)
            y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
            y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[10:]], dtype=object)))
        else:
            train_idx = 20 * len(loss_train) // len(loss_test)
            y_min = np.min(np.min(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
            y_max = np.max(np.max(np.array([loss_train[train_idx:], loss_test[20:]], dtype=object)))
            
        if y_min > 0:
            if y_max > 0:
                axs[3+i].set_ylim([y_min*0.9, y_max*1.1])
            else:
                axs[3+i].set_ylim([y_min*0.9, y_max*0.9])
        else:
            if y_max > 0:
                axs[3+i].set_ylim([y_min*1.1, y_max*1.1])
            else:
                axs[3+i].set_ylim([y_min*1.1, y_max*0.9])
                
        axs[3+i].set_ylabel('loss', fontproperties=axislabelfont)

    axs[4].set_xlabel('epoch', fontproperties=axislabelfont)
    axs[4].set_xlim([1,len(logsig)+1])

    fig.tight_layout()
    fig.savefig(save_name)
    plt.close()

def plot_correlation_plots(model, doc, epoch):
    # mu & sigma correlation plots
    model.eval()
    mus = np.array([])
    sigmas = np.array([])
    mu_shapes = []

    for name, parameter in model.named_parameters():
        if "mu_w" in name:
            mu_shapes.append(parameter.detach().cpu().numpy().shape)
            mus = np.append(mus, parameter.detach().cpu().numpy().flatten())
        if "logsig" in name:
            sigmas = np.append(sigmas, parameter.detach().cpu().numpy().flatten())
            
    bias = np.array([])
    i = 0
    for name, parameter in model.named_parameters():        
        if "bias" in name:
            # print(parameter.detach().cpu().numpy().shape, mu_shapes[i])
            added = (parameter.detach().cpu().numpy()[0]+np.zeros(mu_shapes[i]))
            # print(mu_shapes[i], added.shape)
            bias = np.append(bias, added.flatten())
            i += 1



    plt.figure(dpi=300)
    plt.plot(np.abs(mus), sigmas, lw=0, marker=",")
    plt.title(r"$\mu$ and $\sigma$ correlations (logplot)")
    plt.xscale("log")
    plt.xlabel(r"$|\mu|$")
    plt.ylabel(r"log($\sigma$)")
    plt.xlim(1.e-9, 5.e1)
    plt.savefig(doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "correlation_plots", "mu_sigma_correlation_log.pdf")))
    plt.close()

    plt.figure(dpi=300)
    plt.plot(mus, sigmas, lw=0, marker=",")
    plt.title(r"$\mu$ and $\sigma$ correlations")
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"log($\sigma$)")
    plt.xlim(-2, 2)
    plt.savefig(doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "correlation_plots", "mu_sigma_correlation.pdf")))
    plt.close()

    plt.show()
    plt.figure(dpi=300)
    plt.plot(bias, sigmas, lw=0, marker=".", markersize=1)
    plt.title(r"$\mu$ and $\sigma$ correlations")
    # plt.xscale("log")
    plt.xlabel(r"$bias$")
    plt.ylabel(r"log($\sigma$)")
    plt.savefig(doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "correlation_plots", "bias_sigma_correlation.pdf")))
    plt.close()

def plot_logsigma_development(model, doc, test_loader, epoch, num_rand=30):
    
    model.eval()
    
    with torch.no_grad():
        for i in range(num_rand):
            model.reset_random()
            if i == 0:
                likelihoods = [model.log_prob(test_loader.data, test_loader.cond).detach().cpu().numpy()]
            likelihoods = np.append(likelihoods, [model.log_prob(test_loader.data, test_loader.cond).detach().cpu().numpy()], axis=0) 
    
    likelihood_sigmas = np.std(likelihoods, axis=0)
    # likelihood_sigmas[likelihood_sigmas<1.0e-8] = 1.0e-8

    mus = np.array([])
    log_sigmas = np.array([])
    for name, parameter in model.named_parameters():
        if "mu_w" in name:
            mus = np.append(mus, parameter.detach().cpu().numpy().flatten())
        if "logsig" in name:
            log_sigmas = np.append(log_sigmas, parameter.detach().cpu().numpy().flatten())
            
            
    np.random.shuffle(log_sigmas)


    fig,ax = plt.subplots(dpi=300)
    ax.set_title(f"Epoch {epoch}")
    ax.hist(np.log(likelihood_sigmas**2), bins=100, color="green", alpha=0.5, label="sigmas (loglikelihood)")
    ax.hist(log_sigmas, bins=100, color="orange", alpha=0.5, label="sigmas (parameters)")
    ax.set_yscale("log")
    ax.axvline(np.mean(np.log(likelihood_sigmas**2)), color="green", ls="--", lw=0.5, alpha=1)
    ax.axvline(np.mean(log_sigmas), color="orange", ls="--", lw=0.5, alpha=1)
    ax.set_xlim(-30, 10)
    ax.legend()
    fig.savefig(doc.get_file(os.path.join("plots", f"epoch_{epoch:03d}", "sigma_development.pdf")))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='Where to find the results and save the plots')
    parser.add_argument('--reference_file', help='Where to find the reference data')
    parser.add_argument('--include_coro', action='store_true', help='Also plot the pixel to pixel correlation (Computationally expensive)')
    parser.add_argument('--layer', default=None, type=int, help='Which layer to plot')
    args = parser.parse_args()

    plot_all_hist(args.results_dir, args.reference_file, args.include_coro, args.layer)

if __name__=='__main__':
    main()
