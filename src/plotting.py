import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from calc_obs import *

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
axislabelfont.set_size(24)

tickfont = FontProperties()
tickfont.set_family('serif')
tickfont.set_name('Times New Roman')
tickfont.set_size(22)

def plot_hist(
        file_name,
        data,
        reference,
        axis_label=None,
        xscale='linear',
        yscale='log',
        vmin=None,
        vmax=None,
        n_bins=100):
    data = data[np.isfinite(data)]
    reference = reference[np.isfinite(reference)]

    if vmin is None:
        vmin = min(np.min(data), np.min(reference))
    if vmax is None:
        vmax = max(np.max(data), np.max(reference))
    if xscale=='log':
        if vmin==0:
            vmin = min(np.min(data[data>1e-7]), np.min(reference[reference>1e-7]))
        bins = np.logspace(np.log10(vmin), np.log10(vmax), n_bins)
    else:
        bins = np.linspace(vmin, vmax, n_bins)

    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ax.hist(data, bins=bins, histtype='step', linewidth=2,
        alpha=1, color='blue', density='True', label='CaloINN')

    ax.hist(reference, bins=bins, histtype='stepfilled',
            alpha=0.2, color='blue', density='True', label='CaloINN')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)

    ax.set_xlim([vmin,vmax])

    if axis_label:
        ax.set_xlabel(axis_label, fontproperties=axislabelfont)

    plt.xticks(fontproperties=tickfont)
    plt.yticks(fontproperties=tickfont)

    fig.tight_layout()
    fig.savefig(file_name, bbox_inches='tight')

    plt.close()

def plot_loss(
        file_name,
        loss_train,
        loss_test):
    fig, ax = plt.subplots(1,1,figsize=(12,8), dpi=300)

    c = len(loss_test)/len(loss_train)
    ax.plot(np.arange(c,len(loss_test)+c,c), loss_train, color='blue', label='train loss')
    ax.plot(np.arange(1,len(loss_test)+1), loss_test, color='red', label='test loss')
    ax.legend(loc='upper right', prop=labelfont)

    ax.set_xlim([0,len(loss_test)])
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

def plot_all_hist(results_dir, reference_file, include_coro=False, mask=0, epoch=None):
    data_file = os.path.join(results_dir, 'samples.hdf5')
    if epoch:
        plot_dir = os.path.join(results_dir, 'plots', f'epoch_{epoch:03d}')
    else:
        plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plots = [
        (calc_e_ratio, 'e_ratio.pdf', {}, {'axis_label': r'\(E_{tot}/E_{part}\)'}),
        (calc_e_ratio, 'e_ratio_log.pdf', {}, {'axis_label': r'\(E_{tot}/E_{part}\)', 'xscale': 'log'}),
        (calc_e_detector, 'e_detector.pdf', {}, {'axis_label': r'\(E_{tot}\) (GeV)'}),

        (calc_layer_diff, 'eta_diff_0_1.pdf', {'layer2': 1, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_1\right>-\left<\eta_0\right>\)'}),
        (calc_layer_diff, 'eta_diff_0_2.pdf', {'layer2': 2, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_0\right>\)'}),
        (calc_layer_diff, 'eta_diff_1_2.pdf', {'layer1': 1, 'layer2': 2, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_1\right>\)'}),

        (calc_layer_diff, 'phi_diff_0_1.pdf', {'layer2': 1, 'dir': 'phi'},
            {'axis_label': r'\(\left<\phi_1\right>-\left<\phi_0\right>\)'}),
        (calc_layer_diff, 'phi_diff_0_2.pdf', {'layer2': 2, 'dir': 'phi'},
            {'axis_label': r'\(\left<\phi_2\right>-\left<\phi_0\right>\)'}),
        (calc_layer_diff, 'phi_diff_1_2.pdf', {'layer1': 1, 'layer2': 2, 'dir': 'phi'},
            {'axis_label': r'\(\left<\phi_2\right>-\left<\phi_1\right>\)'})
    ]

    for layer in [0,1,2]:
        plots.append( (calc_e_layer_normd, f'e_normd_layer_{layer}.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}/E_{{tot}}\\)'}) )
        plots.append( (calc_e_layer_normd, f'e_normd_layer_{layer}_log.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}/E_{{tot}}\\)', 'xscale': 'log', 'yscale': 'log', 'vmin': (None, None, 1e-5)[layer]}) )

        plots.append( (calc_e_layer, f'e_layer_{layer}.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}\\) (GeV)'}) )
        plots.append( (calc_e_layer, f'e_layer_{layer}_log.pdf', {'layer': layer},
            {'axis_label': f'\\(E_{layer}\\) (GeV)', 'xscale': 'log', 'yscale': 'log', 'vmin': (None, None, 1e-5)[layer]}) )

        plots.append( (calc_sparsity, f'sparsity_{layer}.pdf', {'layer': layer},
            {'axis_label': f'sparsity layer {layer}', 'xscale': 'linear', 'yscale': 'linear', 'n_bins': 72+1, 'vmin': 0., 'vmax': 1.}) )

        for dir in ['eta', 'phi']:
            plots.append( (calc_centroid_mean, f'{dir}_{layer}.pdf', {'layer': layer, 'dir': dir},
                {'axis_label': f'\\(\\left<\\{dir}_{layer}\\right>\\)', 'yscale': 'log'}) )
            plots.append( (calc_centroid_std, f'{dir}_{layer}_std.pdf', {'layer': layer, 'dir': dir},
                {'axis_label': f'std \\(\\{dir}_{layer}\\)', 'xscale': 'log', 'yscale': 'log'}) )

        for N in range(1,6):
            plots.append( (calc_brightest_voxel, f'{N}_brightest_voxel_layer_{layer}.pdf', {'layer': layer, 'N': N},
                {'axis_label': f'{N}. brightest voxel in layer {layer}', 'yscale': 'linear'}) )

        plots.append( (calc_spectrum, f'spectrum_{layer}.pdf', {'layer': layer},
            {'axis_label': f'voxel energy (GeV)', 'xscale': 'log', 'yscale': 'log'}) )

        if include_coro:
            plots.append( (calc_coro, f'coro02_{layer}.pdf', {'layer': layer},
                {'axis_label': f'\\(C_{{0.2}}\\) layer {layer}', 'xscale': 'linear', 'yscale': 'log'}) )

    data = load_data(data_file)
    reference = load_data(reference_file)

    if mask==1:
        reference = apply_mask(reference, calc_sparsity(reference, layer=1) < 0.1+0.2*np.log10(calc_e_parton(reference)))
    elif mask==2:
        reference = apply_mask(reference, calc_sparsity(reference, layer=1) >= 0.1+0.2*np.log10(calc_e_parton(reference)))
    if mask:
        print(len(reference['energy']))

    for function, name, args1, args2 in plots:
        data_coppy = {k: np.copy(v) for k, v in data.items()}
        reference_coppy = {k: np.copy(v) for k, v in reference.items()}
        plot_hist(
            file_name=os.path.join(plot_dir, name),
            data=function(data_coppy, **args1),
            reference=function(reference_coppy, **args1),
            **args2
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='Where to find the results and save the plots')
    parser.add_argument('--reference_file', help='Where to find the reference data')
    parser.add_argument('--include_coro', action='store_true', help='Also plot the pixel to pixel correlation (Computationally expensive)')
    parser.add_argument('--mask', default=0, type=int)
    args = parser.parse_args()

    plot_all_hist(args.results_dir, args.reference_file, args.include_coro, args.mask)

if __name__=='__main__':
    main()
