import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas as pd
from matplotlib import cm
# from matplotlib.transforms import Bbox

import data_util
import math
import torch


labelfont = FontProperties()
labelfont.set_family('serif')
labelfont.set_size(20)

axislabelfont = FontProperties()
axislabelfont.set_family('serif')
axislabelfont.set_size(20)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_size(20)

rect_double    = (0.05, 0.12, 0.98, 0.97) # left, bottom, right, top
rect_double_with_legend = (0.14, 0.12, 0.98, 0.97) # left, bottom, right, top

     
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
    
    loss_train = np.nan_to_num(loss_train, nan=0, posinf=0, neginf=0)
    loss_test = np.nan_to_num(loss_test, nan=0, posinf=0, neginf=0)
    
    if len(loss_test) <= 10 or (not skip_epochs):
        y_min = np.min( [ np.min(loss_train), np.min(loss_test) ] )
        y_max = np.max( [ np.max(loss_train), np.max(loss_test) ] )
    elif len(loss_test) <= 20:
        train_idx = 10 * len(loss_train) // len(loss_test)
        y_min = np.min( [ np.min(loss_train[train_idx:]), np.min(loss_test[10:]) ] )
        y_max = np.max( [ np.max(loss_train[train_idx:]), np.max(loss_test[10:]) ] )
    # elif len(loss_test) <= 20:
    else:
        train_idx = 20 * len(loss_train) // len(loss_test)
        y_min = np.min( [ np.min(loss_train[train_idx:]), np.min(loss_test[20:]) ] )
        y_max = np.max( [ np.max(loss_train[train_idx:]), np.max(loss_test[20:]) ] )
        
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

def plot_latent(samples, results_dir, epoch=None):
    # TODO: Update
    if epoch is not None:
        plot_dir = os.path.join(results_dir, 'latent', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'latent')
    os.makedirs(plot_dir, exist_ok=True)
    
    max_dims = samples.shape[1]
    
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


def calc_shower_mean(x, c, layer_boundaries, layer, coordinates, direction):
    """Computes the mean of the shower in eta or phi direction for a given layer."""
    
    eta = coordinates[0][layer_boundaries[layer]:layer_boundaries[layer+1]]
    phi = coordinates[1][layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    # Get the layer energy
    layer_energy = x[:, layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    # Compute the mean eta    
    if direction == "eta":
        return np.sum(layer_energy * eta, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16)
    
    elif direction == "phi":
        return np.sum(layer_energy * phi, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16)
    
    else:
        raise ValueError("Invalid direction")
    
def calc_shower_std(x, c, layer_boundaries, layer, coordinates, direction):
    """Computes the std of the shower in eta or phi direction for a given layer."""
    
    mean = calc_shower_mean(x, c, layer_boundaries, layer, coordinates, direction)
    
    eta = coordinates[0][layer_boundaries[layer]:layer_boundaries[layer+1]]
    phi = coordinates[1][layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    # Get the layer energy
    layer_energy = x[:, layer_boundaries[layer]:layer_boundaries[layer+1]]
    
    if direction == "eta":
        discriminant = np.sum(layer_energy * eta * eta, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16) - mean**2
        discriminant[discriminant < 0] = 0
        return np.sqrt(discriminant)
    
    elif direction == "phi":
        discriminant = np.sum(layer_energy * phi * phi, axis=-1) / (layer_energy.sum(axis=-1)+1.e-16) - mean**2
        discriminant[discriminant < 0] = 0
        return np.sqrt(discriminant)
    
    else:
        raise ValueError("Invalid direction")

def calc_flat_energy_distribution(x, c, layer_boundaries, layer=None):
    """Computes the energy distribution of the shower"""
    
    # Factor of two to counter the normalization in the preprocess function
    if layer is None:
        return x.flatten()*2
    
    return x[:, layer_boundaries[layer]:layer_boundaries[layer+1]].flatten()*2
 
def calc_energy(x, c, layer_boundaries, layer=None):
    """Computes the energy of a given layer or of the whole shower"""
    
    # Factor of two to counter the normalization in the preprocess function
    if layer is None:
        return np.sum(x, axis=-1)*2
    
    return np.sum(x[:, layer_boundaries[layer]:layer_boundaries[layer+1]], axis=-1)*2

def calc_etot_over_einc(x, c, layer_boundaries):
    """Computes the total energy of the shower over the incident energy"""
    
    return calc_energy(x, c, layer_boundaries, layer=None) / c
    
    
def get_plot_params(layer_boundaries, coordinates, used_layers=None):
    """Returns the plot parameters for the given layer and direction"""
    
    plots = []
    large_scale = 300
    small_scale = 10
    
    for layer in range(len(layer_boundaries)-1):
        
        for vmax, yscale in zip([large_scale, small_scale], ["linear", "log"]):
            
            if used_layers is not None:
                layer_name = used_layers[layer]
            else:
                layer_name = layer
            
            plots.append(
                [calc_energy, 
                f"energy_{layer_name}.pdf",
                {"layer_boundaries": layer_boundaries, "layer": layer},
                {"axis_label": f'$E_{{\\text{{{layer_name}}}}}$', "yscale": yscale}]
                )
            
            plots.append(
                [calc_shower_mean, 
                f"mean_{layer_name}_eta.pdf",
                {"layer_boundaries": layer_boundaries, "layer": layer, "coordinates": coordinates, "direction": "eta"},
                {"axis_label": f'$\\langle \\eta \\rangle_{{\\text{{{layer_name}}}}}$', "vmin": -vmax, "vmax": vmax}]
                )
            
            plots.append(
                [calc_shower_std, 
                f"std_{layer_name}_eta.pdf",
                {"layer_boundaries": layer_boundaries, "layer": layer, "coordinates": coordinates, "direction": "eta"},
                {"axis_label": f'$\\sigma_{{\\eta, \\text{{{layer_name}}}}}$', "vmin": 0, "vmax": vmax}]
                )
            
            plots.append(
                [calc_shower_mean, 
                f"mean_{layer_name}_phi.pdf",
                {"layer_boundaries": layer_boundaries, "layer": layer, "coordinates": coordinates, "direction": "phi"},
                {"axis_label": f'$\\langle \\phi \\rangle_{{\\text{{{layer_name}}}}}$', "vmin": -vmax, "vmax": vmax}]
                )
            
            plots.append(
                [calc_shower_std, 
                f"std_{layer_name}_phi.pdf",
                {"layer_boundaries": layer_boundaries, "layer": layer, "coordinates": coordinates, "direction": "phi"},
                {"axis_label": f'$\\sigma_{{\\phi, \\text{{{layer_name}}}}}$', "vmin": 0, "vmax": vmax}]
                )
            
    plots.append(
        [calc_flat_energy_distribution, 
         "flat_energy_distribution.pdf",
         {"layer_boundaries": layer_boundaries},
         {"axis_label": r'$Voxel distribution$', "n_bins": 30}]
        )
    
    plots.append(
        [calc_flat_energy_distribution, 
         "flat_energy_distribution.pdf",
         {"layer_boundaries": layer_boundaries},
         {"axis_label": r'$Voxel distribution$', "yscale": "log", "xscale": "log", "vmin": 0.1, "n_bins": 30}]
        )
    
    plots.append(
        [calc_etot_over_einc, 
         "etot_over_einc.pdf",
         {"layer_boundaries": layer_boundaries},
         {"axis_label": r'$E_{tot} / E_{inc}$'}]
        )        

    return plots

def plot_hist(
        file_name,
        data,
        reference,
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
        panel_scale="linear",
        panel_range=[0.8, 1.2],
        density=True,
        labels=None,
        errorbars_true=False,
        errorbars_fake=False,
        y_label=True, 
        fig = None,
        print_means=False,):
    
    

    if type(errorbars_fake) == bool:
        errorbars_fake = [errorbars_fake]
    
    if type(data)==list and type(data[0])==np.ndarray:
        data_list = data
    else:
        data_list = [data]
             
    if len(errorbars_fake) != len(data_list):
        assert len(errorbars_fake) == 1, "Wrong size for the errorbars index"
        errorbars_fake = [errorbars_fake[0] for _ in range(len(data_list))]
    
    for i in range(len(data_list)):
        finite = np.isfinite(data_list[i])
        data_list[i] = data_list[i][finite]
        
    
    finite = np.isfinite(reference)
    reference = reference[finite]

    all_data = [reference] + data_list

    try:
        # Set the plotting boundaries
        if vmin is None:
            vmin = np.inf
            for elem in all_data:
                vmin = np.min([np.min(elem), vmin])
        if vmax is None:
            vmax = -np.inf
            for elem in all_data:
                vmax = np.max([np.max(elem), vmax])
                
        if vmax == vmin:
            vmax += 0.0001
            vmin -= 0.0001
    except:
        print("Error in setting the boundaries")
        print(axis_label)
        print(all_data)
        return
            
    # Get the bins (Modifications needed if logscale is used)
    if xscale=='log':
        
        if vmin<=0:
            vmin = np.inf     
            for elem in all_data:
                vmin = np.min([np.min(elem[elem>1e-7]), vmin])
                
        if isinstance(n_bins, int):
            bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins)
        else:
            bins = n_bins
    else:
        if isinstance(n_bins, int):
            bins = np.linspace(vmin, vmax, n_bins)
        else:
            bins = n_bins
    

    color = 'blue'
        
    create_fig = False
    if ax is None:
        create_fig = True
        fig, ax = plt.subplots(1,1,figsize=(6,6))    
        
        
    # Plot the reference data
    if not errorbars_true:
        ns_true, bins_true, _ = ax.hist(reference, bins=bins, histtype='stepfilled',
                alpha=0.5, color=color, density=density, label='GEANT', linewidth=1.5)
    else:
        dup_last = lambda a: np.append(a, a[-1])

        bins_true = bins
        
        counts, _ = np.histogram(reference, bins_true, density=False)
        
        ns_true, _ = np.histogram(reference, bins_true, density=density)
        
        mask = (counts == 0)
        counts[mask] = 1
        
        if density: # relative error stays the same
            ref_err = ns_true / np.sqrt(counts)
            
        else:
            ref_err = np.sqrt(ns_true)
        
        
        ref_err[mask] = 0
        
        ax.step(bins_true, dup_last(ns_true), color="blue", alpha=1,
                        linewidth=1.5, where='post', label='GEANT')
        
        ax.step(bins_true, dup_last(ns_true - ref_err), color="blue", alpha=0.5,
                        linewidth=0.5, where='post')
        ax.step(bins_true, dup_last(ns_true + ref_err), color="blue", alpha=0.5,
                        linewidth=0.5, where='post')

        ax.fill_between(bins_true, dup_last(ns_true - ref_err), dup_last(ns_true + ref_err), 
                        facecolor="blue", alpha=0.3, step='post')
    
    # Plot the generated data
    alt_colors = ["green", "red", "orange", "pink", "black"]
    
    ns_fakes = []
    bins_fakes = []
    
    for i, data in enumerate(data_list):
        if not errorbars_fake[i]:
            
            # Modify the labels
            if labels is None:
                label = "VAE"
            else:
                label = labels[i]
            
            ns_i, bins_i, _ = ax.hist(data, bins=bins, histtype='step', linewidth=1.5,
                alpha=1, density=density, label=label, color=alt_colors[i])
    
    
            ns_fakes.append(ns_i)
            bins_fakes.append(bins_i)
            
        else:
            # labels
            if labels is None:
                label = "VAE"
            else:
                label = labels[i]
                
            data = data_list[i]
            
            dup_last = lambda a: np.append(a, a[-1])
            
            bins_i = bins
            
            counts, _ = np.histogram(data, bins_i, density=False)
            
            ns_i, _ = np.histogram(data, bins_i, density=density)
            
            
            mask = (counts == 0)
            counts[mask] = 1
            if density: # relative error stays the same
                data_err = ns_i / np.sqrt(counts)
                
            else:
                data_err = np.sqrt(ns_i)
                
                
            data_err[mask] = 0
            
            ax.step(bins_i, dup_last(ns_i), color=alt_colors[i], alpha=1,
                            linewidth=1.5, where='post', label=label)
            
            ax.step(bins_i, dup_last(ns_i - data_err), color=alt_colors[i], alpha=0.5,
                            linewidth=0.5, where='post')
            ax.step(bins_i, dup_last(ns_i + data_err), color=alt_colors[i], alpha=0.5,
                            linewidth=0.5, where='post')

            ax.fill_between(bins_i, dup_last(ns_i - data_err), dup_last(ns_i + data_err), 
                            facecolor=alt_colors[i], alpha=0.3, step='post')
            
            
            ns_fakes.append(ns_i)
            bins_fakes.append(bins_i)
    
    if y_label:              
        ax.set_ylabel(r"$Normalized counts$")
        
    if panel_ax is not None:
        
        for i in range(len(bins_fakes)):
            assert len(bins_true) == len(bins_fakes[i]), f"Length of bins_true: {len(bins_true)}, Length of bins_fakes[{i}]: {len(bins_fakes[i])}"
            # assert (bins_true - bins_fakes[i] < 1.e-7).all()
        
        for i, (ns_reco, bins_reco) in enumerate(zip(ns_fakes, bins_fakes)):
            
            if labels is not None:
                label = labels[i]
            else:
                label = "VAE"
            
            if i==0:
                mask = ns_true == 0
                ns_true[mask] = 1
                
            panel_data = ns_reco/ns_true
            
            panel_data[mask] = 0
            
            widths = 1.2*(bins_true[1:] - bins_true[:-1])
            panel_ax.axhline(1, color="black", ls="--", alpha=0.5, lw=1
                             )
            panel_ax.hist(bins_reco[:-1], bins_reco[1:]-widths, weights=panel_data, histtype="step", lw=1, ls="--",
                          label=f'{label}/GEANT', color=alt_colors[i])
            
            if y_label:
                panel_ax.set_ylabel(r"$\frac{{Model}}{{GEANT}}$")
        
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if panel_ax is not None:
        panel_ax.set_yscale(panel_scale)
        panel_ax.set_xscale(xscale)

    ax.set_xlim([vmin,vmax])
    if panel_ax is not None:
        panel_ax.set_xlim([vmin,vmax])
        panel_ax.set_ylim(panel_range)
        
    if ymin is not None or ymax is not None:
        ax.set_ylim((ymin, ymax))
        
    if panel_ax is not None:
        lower_bound, upper_bound = ax.get_ylim()
        
        ticks = ax.get_yticks()
        ticks = ticks[ticks >= lower_bound]
        ticks = ticks[ticks <= upper_bound]
        
        if yscale != "log":
            ticks = ticks[1:]

        ax.set_yticks(ticks)
        
        
    if print_means and fig is not None:
        legend = ax.legend(["Data", "Data", "Data"], loc="best")
        plt.draw()
        bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        legend.remove()
        
        # print(bbox.x0, bbox.y0)
        
        text = ""
        for i, data in enumerate(all_data):
            mu = np.mean(data)
            std = np.std(data)
            
            if i == 0:
                mu_0 = mu
            
            # text += f'$\mu$={mu:.3e}$\pm${std:.1e}\n'
            # text += f'$\mu$={mu:.3e}\n'
            
            if mu_0 == 0:
                break
            
            text += f'{mu / mu_0:0.3f} \\pm {std / mu_0 / np.sqrt(len(data)):0.3f}\n'
        
        ax.text(bbox.x0, bbox.y0, text, transform=fig.transFigure, fontsize=15)
    

    if axis_label is not None:
        if panel_ax is None:
            ax.set_xlabel(axis_label)
        else:
            panel_ax.set_xlabel(axis_label)
    

    if create_fig:
        fig.tight_layout()
        fig.savefig(file_name, bbox_inches='tight')
        plt.close()
 
def plot_all_hist(xs, cs, plot_params, plot_dir=None, single_plots=False, summary_plot=False, summary_plot_name=None, labels=None, 
                  errorbars_true=False, errorbars_fake=False, plots_per_row=5, plot_seperate_legend=True, ncol=None,
                  print_means=False):

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    plots = plot_params

    if single_plots and plot_dir is not None:
        
        for i, (func, name, args1, args2) in enumerate(plots):
            
            fig, axs = plt.subplots(2,1, dpi=300, figsize=(7,6*1.3), gridspec_kw={'height_ratios': [1, 0.3]})
            
                            
            # Add ylabels to the leftmost plots
            if i % plots_per_row == 0:
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_double_with_legend)
                ylabel = True
            else:
                fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_double)
                ylabel = False
                
            plot_hist(
                file_name=None,
                data=[func(x, c, **args1) for x, c in zip(xs[1:], cs[1:])],
                reference=func(xs[0], cs[0], **args1),
                ax=axs[0],
                panel_ax=axs[1],
                labels=labels,
                errorbars_fake=errorbars_fake,
                errorbars_true=errorbars_true,
                y_label=ylabel,
                fig=fig,
                print_means=print_means,
                **args2)

            
            if not plot_seperate_legend:
                # Add figure legend on every rightmost plot
                if i % plots_per_row == plots_per_row-1 or i == len(plots)-1:
                    # Get legend handles and labels from first axis
                    lines1, labels1 = axs[0].get_legend_handles_labels()

                    all_lines = lines1
                    all_labels = labels1

                    # Create a figure-wide legend
                    fig.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(0.95, 0.98))
                
            # Hide the (shared) x-axis
            axs[0].xaxis.set_visible(False)

            
            fig.subplots_adjust(hspace=0)
            # fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=rect_double)
            fig.savefig(os.path.join(plot_dir, f"{i+1:02}_"+name), dpi=300)
            # fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            # fig.savefig(plot_dir+f"{i+1:02}_"+name, bbox_inches='tight', dpi=300)
            plt.close()

        if plot_seperate_legend:
            fig_leg = plt.figure(figsize=(8., 2./3.)) # if 1 particle, use (8,2) for 3 particles
            ax_leg = fig_leg.add_subplot(111)
            
            # add the legend from the previous axes
            lines1, labels1 = axs[0].get_legend_handles_labels()

            all_lines = lines1
            all_labels = labels1
            entries = len(labels)+1
            
            if ncol is not None:
                ax_leg.legend(all_lines, all_labels, ncol=ncol, loc='center')
            elif entries >= 4:
                ax_leg.legend(all_lines, all_labels, ncol=(entries+1)//2, loc='center')
            else:
                ax_leg.legend(all_lines, all_labels, ncol=entries, loc='center')
            # hide the axes frame and the x/y labels
            ax_leg.axis('off')
            fig_leg.savefig(os.path.join(plot_dir,f"{0:02}_"+"legend.pdf"), bbox_inches='tight', dpi=300, pad_inches=0.1)

            plt.close()

    if not summary_plot:
        return

    # Plot all the histogramms in one file
    number_of_plots = len(plots)
    rows = number_of_plots // plots_per_row
    if number_of_plots%plots_per_row != 0:
        rows += 1
    heights = [1, 0.3, 0.3]*rows

    fig, axs = plt.subplots(rows*3,plots_per_row, dpi=500, figsize=(plots_per_row*7,6*np.sum(heights)), gridspec_kw={'height_ratios': heights})

    iteration = 0
    for i in range(rows*3):
        
        if i%3 == 1:
            iteration -= plots_per_row
            
        for j in range(plots_per_row):
            
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
            func, name, args1, args2 = plots[iteration]
            
            
            if i % 3 == 0:
                # plot the main data
                plot_hist(
                        file_name=None,
                        data=[func(x, c, **args1) for x, c in zip(xs[1:], cs[1:])],
                        reference=func(xs[0], cs[0], **args1),
                        ax=axs[i,j],
                        panel_ax=axs[i+1,j],
                        labels=labels,
                        errorbars_fake=errorbars_fake,
                        errorbars_true=errorbars_true,
                        y_label=iteration % plots_per_row == 0,
                        fig=fig,
                        print_means=print_means,
                        **args2)
                
                # Hide the (shared) x-axis
                axs[i,j].xaxis.set_visible(False)
                
                
                if i == 0 and j==0:
                    # Get legend handles and labels from first axis
                    lines1, labels1 = axs[0,0].get_legend_handles_labels()

                    # Get legend handles and labels from second axis
                    lines2, labels2 = axs[1,0].get_legend_handles_labels()

                    # Combine handles and labels from both axes
                    all_lines = lines1 + lines2
                    all_labels = labels1 + labels2

                    # Create a figure-wide legend
                    fig.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1,0.5))
                     
                iteration += 1

            if i % 3 == 1:
                iteration += 1             

    fig.subplots_adjust(hspace=0)
    if plot_dir is not None and summary_plot_name is not None:
        fig.savefig(os.path.join(plot_dir,summary_plot_name), bbox_inches='tight', dpi=500)
        plt.close()
    else:
        plt.show()
      
        
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
