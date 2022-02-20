import os
import argparse

import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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

def load_data(data_file):
    full_file = h5py.File(data_file, 'r')
    layer_0 = np.float32(full_file['layer_0'][:] / 1e3)
    layer_1 = np.float32(full_file['layer_1'][:] / 1e3)
    layer_2 = np.float32(full_file['layer_2'][:] / 1e3)
    energy = np.float32(full_file['energy'][:] /1e0)
    full_file.close()

    data = {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy[:,0]
    }

    return data

def plot_hist(
        file_name,
        data,
        reference=None,
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

    if reference is not None:
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

def calc_e_layer(data, layer=0):
    return np.sum(data[f'layer_{layer}'],axis=(1,2))

def calc_e_detector(data):
    return np.sum(data['layer_0'],(1,2)) + np.sum(data['layer_1'],(1,2)) + np.sum(data['layer_2'],(1,2))

def calc_e_relation(data):
    e_detector = calc_e_detector(data)
    e_parton = data['energy']
    return e_detector/e_parton

def calc_e_layer_normd(data, layer=0):
    e_layer = calc_e_layer(data, layer)
    e_total = calc_e_detector(data)
    return e_layer/e_total

def calc_brightest_voxel(data, layer=0, N=1):
    layer = data[f'layer_{layer}']
    layer = layer.reshape((layer.shape[0], -1))
    layer = layer[layer.sum(axis=1)>0]
    layer.sort(axis=1)
    return layer[:,-N]/layer.sum(axis=1)

def calc_centroid(data, layer=0, dir='phi'):
    if dir == 'phi':
        cells = (3, 12, 12)[layer]
    elif dir == 'eta':
        cells = (96, 12, 6)[layer]
    else:
        raise ValueError(f"dir={dir} not in ['eta', 'phi']")
    bins = np.linspace(-240, 240, cells + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2.

    layer = data[f'layer_{layer}']
    energies = layer.sum(axis=(1, 2))

    if dir == 'phi':
        value = bin_centers.reshape(-1, 1)
    elif dir == 'eta':
        value = bin_centers.reshape(1, -1)

    mean = np.sum(layer * value, axis=(1, 2))/(energies+1e-10)
    std = np.sqrt(np.sum((layer * value - mean[:,None, None])**2, axis=(1, 2))/(energies+1e-10))

    return mean, std

def calc_centroid_mean(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[0]

def calc_centroid_std(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[1]

def calc_layer_diff(data, layer1=0, layer2=1, dir='phi'):
    mean_1, _ = calc_centroid(data, layer1, dir)
    mean_2, _ = calc_centroid(data, layer2, dir)
    return mean_1 - mean_2

def calc_sparsity(data, layer=0, threshold=1e-5):
    layer = data[f'layer_{layer}']
    return (layer > threshold).mean((1, 2))

def plot_all_hist(results_dir, reference_file):
    data_file = os.path.join(results_dir, 'samples.hdf5')
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plots = [
        (calc_e_relation, 'e_relation.pdf', {}, {'axis_label': r'\(E_{tot}/E_{part}\)'}),
        (calc_e_detector, 'e_detector.pdf', {}, {'axis_label': r'\(E_{tot}\) (GeV)'}),

        (calc_layer_diff, 'eta_diff_0_1.pdf', {'layer2': 1, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_1\right>-\left<\eta_0\right>\)'}),
        (calc_layer_diff, 'eta_diff_0_2.pdf', {'layer2': 2, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_0\right>\)'}),
        (calc_layer_diff, 'eta_diff_1_2.pdf', {'layer2': 1, 'layer2': 2, 'dir': 'eta'},
            {'axis_label': r'\(\left<\eta_2\right>-\left<\eta_1\right>\)'}),

        (calc_layer_diff, 'phi_diff_0_1.pdf', {'layer2': 1, 'dir': 'phi'},
            {'axis_label': r'\(\left<\phi_1\right>-\left<\phi_0\right>\)'}),
        (calc_layer_diff, 'phi_diff_0_2.pdf', {'layer2': 2, 'dir': 'phi'},
            {'axis_label': r'\(\left<\phi_2\right>-\left<\phi_0\right>\)'}),
        (calc_layer_diff, 'phi_diff_1_2.pdf', {'layer2': 1, 'layer2': 2, 'dir': 'phi'},
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

    data = load_data(data_file)
    reference = load_data(reference_file)

    for function, name, args1, args2 in plots:
        plot_hist(
            file_name=os.path.join(plot_dir, name),
            data=function(data, **args1),
            reference=function(reference, **args1),
            **args2
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', help='Where to find the results and save the plots')
    parser.add_argument('--reference_file', help='Where to find the reference data')
    args = parser.parse_args()

    plot_all_hist(args.results_dir, args.reference_file)

if __name__=='__main__':
    main()
