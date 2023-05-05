# pylint: disable=invalid-name
""" Helper functions that compute the high-level observables of calorimeter showers.

    Used for

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

    "CaloFlow II: Even Faster and Still Accurate Generation of Calorimeter Showers with
     Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2110.11377

    Functions inspired by
    "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-LayerElectromagnetic
     Calorimeters with Generative Adversarial Networks"
    by Michela Paganini, Luke de Oliveira, and Benjamin Nachman
    arxiv:1712.10321
    https://github.com/hep-lbdl/CaloGAN
"""

import numpy as np

# number of voxel per layer in the given dimension, see fig 2 of 1712.10321
PHI_CELLS = {0: 3, 1: 12, 2: 12}
ETA_CELLS = {0: 96, 1: 12, 2: 6}

def to_np_thres(tensor, threshold):
    """ moves tensor to CPU, then to numpy, then applies threshold """
    ret = tensor.clamp_(0., 1e5).to('cpu').numpy()
    #ret = tensor.to('cpu').numpy()
    ret = np.where(ret < threshold, np.zeros_like(ret), ret)
    return ret

def layer_split(data):
    """ splits data into the 3 layers """
    return np.split(data, [288, 432], axis=-1)

def layer_std(layer1, layer2, total):
    """ helper function for standard deviation of layer depth"""
    term1 = (layer1 + 4.*layer2) / total
    term2 = ((layer1 + 2.*layer2)/total)**2
    return np.sqrt(term1-term2)

def energy_sum(data, normalization=1e3):
    """ Returns the energy sum of all energy depositions.
        If len(data.shape) is 3, the sum is taken over the last 2 axis (summing voxels per layer).
        If it is 2, the sum is taken over the last axis (summing voxels of entire event).
        normalization of 1e3 accounts for MeV to GeV unit conversion.
    """
    if len(data.shape) == 3:
        ret = data.sum(axis=(1, 2))
    else:
        ret = data.sum(axis=-1)
    return ret / normalization

def energy_ratio(data, layer):
    """ Returns the sum of energy of the given layer divided by the total energy """
    ret = energy_sum(layer_split(data)[layer]) / energy_sum(data)
    return ret

def layer_sparsity(data, threshold, layer=None):
    """ Returns the sparsity (=fraction of voxel above threshold) of the given layer
        Supports either sparsity in given data array (of size 3), assuming it's a layer
        or needs a layer_nr if data has size 2 (assuming it's a full shower).
    """
    if len(data.shape) == 3:
        sparsity = (data > threshold).mean((1, 2))
    else:
        if layer is not None:
            data = layer_split(data)[layer]
        sparsity = (data > threshold).mean(-1)
    return sparsity

def n_brightest_voxel(data, num_brightest, ratio=True):
    """ Returns the ratio of the "num_brightest" voxel to the energy deposited in the layer. """
    if len(data.shape) == 3:
        data = data.reshape(len(data), -1)
    top_n = np.sort(data, axis=1)[:, -max(num_brightest):]
    energies = data.sum(axis=-1).reshape(-1, 1)
    ret = top_n[:, [-num for num in num_brightest]]
    if ratio:
        # why did I filter energies>0?
        #ret = (ret/ (energies + 1e-16))[np.tile(energies > 0, len(num_brightest))].reshape(
        #    -1, len(num_brightest))
        ret = (ret/ (energies + 1e-16)).reshape(-1, len(num_brightest))
    return ret

def ratio_two_brightest(data):
    """ Returns the ratio of the difference of the 2 brightest voxels to their sum. """
    top = np.sort(data, axis=1)[:, -2:]
    #ret = ((top[:, 1] - top[:, 0]) / (top[:, 0] + top[:, 1] + 1e-16))[top[:, 1] > 0]
    ret = ((top[:, 1] - top[:, 0]) / (top[:, 0] + top[:, 1] + 1e-16))
    return ret

def maxdepth_nr(data):
    """ Returns the layer that has the last energy deposition """
    _, layer_1, layer_2 = layer_split(data)

    maxdepth = 2* (energy_sum(layer_2) != 0)
    maxdepth[maxdepth == 0] = 1* (energy_sum(layer_1[maxdepth == 0]) != 0)
    return maxdepth

def depth_weighted_energy(data):
    """ Returns the depth-weighted total energy deposit in all 3 layers for a given batch. """
    _, layer_1, layer_2 = layer_split(data)
    ret = energy_sum(layer_1, normalization=1.) + 2.* energy_sum(layer_2, normalization=1.)
    return ret

def depth_weighted_energy_normed_std(data):
    """ Returns the standard deviation of the depth-weighted total energy deposit in all 3 layers
        normalized by the total deposited energy for a given batch
    """
    _, layer_1, layer_2 = layer_split(data)
    layer1 = energy_sum(layer_1, normalization=1.)
    layer2 = energy_sum(layer_2, normalization=1.)
    energies = energy_sum(data, normalization=1.)
    return layer_std(layer1, layer2, energies)

def center_of_energy(data, layer, direction):
    """ Returns the center of energy in the direction 'direction' for layer 'layer'. """
    if direction == 'eta':
        bins = np.linspace(-240, 240, ETA_CELLS[layer] + 1)
    elif direction == 'phi':
        bins = np.linspace(-240, 240, PHI_CELLS[layer] + 1)
    else:
        raise ValueError("direction={} not in ['eta', 'phi']".format(direction))
    bin_centers = (bins[1:] + bins[:-1]) / 2.

    if direction == 'phi':
        data = data.reshape(len(data), PHI_CELLS[layer], -1)
        ret = (data * bin_centers.reshape(-1, 1)).sum(axis=(1, 2))
    else:
        data = data.reshape(len(data), -1, ETA_CELLS[layer])
        ret = (data * bin_centers.reshape(1, -1)).sum(axis=(1, 2))
    energies = energy_sum(data, normalization=1.)
    ret = ret / (energies + 1e-8)
    return ret

def center_of_energy_std(data, layer, direction):
    """ Returns the standard deviation of center of energy in the direction
        'direction' for layer 'layer'.
    """
    if direction == 'eta':
        bins = np.linspace(-240, 240, ETA_CELLS[layer] + 1)
    elif direction == 'phi':
        bins = np.linspace(-240, 240, PHI_CELLS[layer] + 1)
    else:
        raise ValueError("direction={} not in ['eta', 'phi']".format(direction))
    bin_centers = (bins[1:] + bins[:-1]) / 2.

    if direction == 'phi':
        data = data.reshape(len(data), PHI_CELLS[layer], -1)
        ret = (data * bin_centers.reshape(-1, 1)).sum(axis=(1, 2))
        ret2 = (data * bin_centers.reshape(-1, 1)**2).sum(axis=(1, 2))
        print(ret[np.isnan(ret)].shape, ret2[np.isnan(ret2)].shape)
    else:
        data = data.reshape(len(data), -1, ETA_CELLS[layer])
        ret = (data * bin_centers.reshape(1, -1)).sum(axis=(1, 2))
        ret2 = (data * bin_centers.reshape(1, -1)**2).sum(axis=(1, 2))
    energies = energy_sum(data, normalization=1.)
    
    ret = np.sqrt((ret2 / (energies+1e-8)) - (ret / (energies+1e-8)) ** 2)
    return ret
