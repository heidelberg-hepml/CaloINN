import numpy as np
import h5py
import torch

from myDataLoader import MyDataLoader

def load_data(data_file, use_extra_dim=False,threshold=1e-7, mask=0):
    full_file = h5py.File(data_file, 'r')
    layer_0 = np.float32(full_file['layer_0'][:] / 1e5)
    layer_1 = np.float32(full_file['layer_1'][:] / 1e5)
    layer_2 = np.float32(full_file['layer_2'][:] / 1e5)
    energy = np.float32(full_file['energy'][:] /1e2)
    full_file.close()

    layer0 = layer_0.reshape(layer_0.shape[0], -1)
    layer1 = layer_1.reshape(layer_1.shape[0], -1)
    layer2 = layer_2.reshape(layer_2.shape[0], -1)

    if mask==1:
        binary_mask = (layer1 > threshold).mean(1) < 0.25
    elif mask==2:
        binary_mask = (layer1 > threshold).mean(1) >= 0.25
    else:
        binary_mask = np.full(len(energy), True)

    x = np.concatenate((layer0, layer1, layer2), 1)[binary_mask]
    c = energy[binary_mask]

    if use_extra_dim:
        x = add_extra_dim(x, c)
    return x, c

def save_data(data_file, samples, energies, threshold=0.01, use_extra_dim=False):
    assert len(energies) == len(samples)

    if use_extra_dim:
        samples = remove_extra_dim(samples, energies)

    data = 1e5*samples.clip(0., 1.)
    data[data < threshold] = 0.

    energies = energies*1e2
    overflow = np.zeros((len(energies), 3))
    layer_0 = data[..., :288].reshape(-1, 3, 96)
    layer_1 = data[..., 288:432].reshape(-1, 12, 12)
    layer_2 = data[..., 432:].reshape(-1, 12, 6)

    save_file = h5py.File(data_file, 'w')

    save_file.create_dataset('layer_0', data=layer_0)
    save_file.create_dataset('layer_1', data=layer_1)
    save_file.create_dataset('layer_2', data=layer_2)
    save_file.create_dataset('energy', data=energies)
    save_file.create_dataset('overflow', data=overflow)

    save_file.close()

def add_extra_dim(data, energies):
    s = np.sum(data, axis=1, keepdims=True)
    factors = s/energies
    data /= s
    return np.concatenate((data, factors), axis=1)

def remove_extra_dim(data, energies):
    factors = data[:,[-1]]
    data = data[:,:-1]
    data /= np.sum(data, axis=1, keepdims=True)
    return data*energies*factors

def get_loaders(data_file, batch_size, ratio=0.8, device='cpu', width_noise=1e-7, use_extra_dim=False, mask=0):
    data, cond = load_data(data_file, use_extra_dim, mask=mask)
    data = torch.tensor(data, device=device)
    cond = torch.tensor(cond, device=device)
    index = torch.randperm(len(data), device=device)
    split = int(ratio*len(data))
    data_train = data[index[:split]]
    data_test = data[index[split:]]
    cond_train = cond[index[:split]]
    cond_test = cond[index[split:]]
    loader_train = MyDataLoader(data_train, cond_train, batch_size, width_noise=width_noise)
    loader_test = MyDataLoader(data_test, cond_test, batch_size, width_noise=width_noise)
    return loader_train, loader_test
