import numpy as np
import h5py
import torch

from myDataLoader import MyDataLoader

def load_data(data_file, mask=None, threshold=1e-5):
    full_file = h5py.File(data_file, 'r')
    layer_0 = full_file['layer_0'][:] / 1e3
    layer_1 = full_file['layer_1'][:] / 1e3
    layer_2 = full_file['layer_2'][:] / 1e3
    energy = full_file['energy'][:] / 1e0
    full_file.close()

    data = {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy
    }

    if mask==1:
        binary_mask = (layer_1 > threshold).mean((1,2)) < 0.1+0.2*np.log10(energy[:,0])
    elif mask==2:
        binary_mask = (layer_1 > threshold).mean((1,2)) >= 0.1+0.2*np.log10(energy[:,0])
    else:
        binary_mask = np.full(len(energy), True)

    return apply_mask(data, binary_mask)

def save_data(data, data_file):
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']
    overflow = np.zeros((len(energy), 3))

    save_file = h5py.File(data_file, 'w')
    save_file.create_dataset('layer_0', data=layer_0*1e3)
    save_file.create_dataset('layer_1', data=layer_1*1e3)
    save_file.create_dataset('layer_2', data=layer_2*1e3)
    save_file.create_dataset('energy', data=energy*1e0)
    save_file.create_dataset('overflow', data=overflow)
    save_file.close()

def apply_mask(data, mask):
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']

    return {
        'layer_0': layer_0[mask],
        'layer_1': layer_1[mask],
        'layer_2': layer_2[mask],
        'energy': energy[mask]
    }

def preprocess(data, use_extra_dim=False, use_extra_dims=False, threshold=1e-5, layer=None):
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']
    
    layer0 = layer_0.reshape(layer_0.shape[0], -1)
    layer1 = layer_1.reshape(layer_1.shape[0], -1)
    layer2 = layer_2.reshape(layer_2.shape[0], -1)

    binary_mask = np.full(len(energy), True)

    if layer is not None:
        x = (layer0, layer1, layer2)[layer]
        binary_mask &= np.sum(x, axis=1) > threshold
    else:
        x = np.concatenate((layer0, layer1, layer2), 1)

    if use_extra_dims:
        binary_mask &= np.sum(x, axis=1) < energy[:,0]

    x = x[binary_mask]
    c = energy[binary_mask]

    if use_extra_dims:
        x = add_extra_dims(x, c)
    elif use_extra_dim:
        x = add_extra_dim(x, c)
    return x, c

def postprocess(samples, energy, use_extra_dim=False, use_extra_dims=False, layer=None, threshold=1e-5):
    assert len(energy) == len(samples)

    if use_extra_dims:
        samples = remove_extra_dims(samples, energy)
    elif use_extra_dim:
        samples = remove_extra_dim(samples, energy)

    samples[samples < threshold] = 0.

    if layer is not None:
        layer_0 = np.zeros((len(samples), 3, 96))
        layer_1 = np.zeros((len(samples), 12, 12))
        layer_2 = np.zeros((len(samples), 12, 6))
        (layer_0, layer_1, layer_2)[layer][:] = samples.reshape(((-1, 3, 96),(-1, 12, 12),(-1, 12, 6))[layer])
    else:
        layer_0 = samples[..., :288].reshape(-1, 3, 96)
        layer_1 = samples[..., 288:432].reshape(-1, 12, 12)
        layer_2 = samples[..., 432:].reshape(-1, 12, 6)

    return {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy
    }

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

def add_extra_dims(data, e_part):
    e0 = np.sum(data[..., :288], axis=1, keepdims=True)
    e1 = np.sum(data[..., 288:432], axis=1, keepdims=True)
    e2 = np.sum(data[..., 432:], axis=1, keepdims=True)
    u1 = (e0+e1+e2)/e_part
    u2 = e0/(e0+e1+e2)
    u3 = e1/(e1+e2+1e-7)
    data /= np.sum(data, axis=1, keepdims=True)
    return np.concatenate((data, u1/(1-u1+1e-7), u2/(1-u2+1e-7), u3/(1-u3+1e-7)), axis=1)

def remove_extra_dims(data, e_part):
    u1 = data[:,[-3]]/(1+data[:,[-3]])
    u2 = data[:,[-2]]/(1+data[:,[-2]])
    u3 = data[:,[-1]]/(1+data[:,[-1]])
    e_tot = u1*e_part
    e0 = u2*e_tot
    e1 = u3*(e_tot-e0)
    e2 = e_tot - e0 -e1
    data = data[:,:-3]
    data[data<0] = 0.
    data[..., :288]    /= (np.sum(data[..., :288], axis=1, keepdims=True) + 1e-7)
    data[..., 288:432] /= (np.sum(data[..., 288:432], axis=1, keepdims=True) + 1e-7)
    data[..., 432:]    /= (np.sum(data[..., 432:], axis=1, keepdims=True) + 1e-7)
    data[..., :288]    *= e0
    data[..., 288:432] *= e1
    data[..., 432:]    *= e2
    return data

# def add_extra_dims(data, energies):
#     factors_0 = np.sum(data[..., :288], axis=1, keepdims=True)/energies
#     factors_1 = np.sum(data[..., 288:432], axis=1, keepdims=True)/energies
#     factors_2 = np.sum(data[..., 432:], axis=1, keepdims=True)/energies
#     data /= np.sum(data, axis=1, keepdims=True)
#     return np.concatenate((data, factors_0, factors_1, factors_2), axis=1)

# def remove_extra_dims(data, energies):
#     factors_0 = data[:,[-3]]
#     factors_1 = data[:,[-2]]
#     factors_2 = data[:,[-1]]
#     data = data[:,:-3]
#     data[data<0] = 0.
#     data[..., :288]    /= (np.sum(data[..., :288], axis=1, keepdims=True) + 1e-7)
#     data[..., 288:432] /= (np.sum(data[..., 288:432], axis=1, keepdims=True) + 1e-7)
#     data[..., 432:]    /= (np.sum(data[..., 432:], axis=1, keepdims=True) + 1e-7)
#     data[..., :288]    *= energies*factors_0
#     data[..., 288:432] *= energies*factors_1
#     data[..., 432:]    *= energies*factors_2
#     return data

def get_loaders(data_file, batch_size, ratio=0.8, device='cpu',
        width_noise=1e-7, use_extra_dim=False, use_extra_dims=False, mask=0, layer=None):
    data, cond = preprocess(load_data(data_file, mask),
        use_extra_dim, use_extra_dims, layer=layer)
    postprocess(data, cond, use_extra_dim, use_extra_dims, layer)
    data = torch.tensor(data, device=device, dtype=torch.get_default_dtype())
    cond = torch.tensor(cond, device=device, dtype=torch.get_default_dtype())
    index = torch.randperm(len(data), device=device)
    split = int(ratio*len(data))
    data_train = data[index[:split]]
    data_test = data[index[split:]]
    cond_train = cond[index[:split]]
    cond_test = cond[index[split:]]
    loader_train = MyDataLoader(data_train, cond_train, batch_size, width_noise=width_noise)
    loader_test = MyDataLoader(data_test, cond_test, batch_size, width_noise=width_noise)
    return loader_train, loader_test
