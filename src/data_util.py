import numpy as np
import h5py
import torch
from copy import deepcopy
from myDataLoader import MyDataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import os
import warnings

def load_data(data_file):
    """load the requested h5py file and return it as dict"""
    full_file = h5py.File(data_file, 'r')
    layer_0 = full_file['layer_0'][:] / 1e3
    layer_1 = full_file['layer_1'][:] / 1e3
    layer_2 = full_file['layer_2'][:] / 1e3
    energy = full_file['energy'][:] / 1e0
    # TODO: Use correct normalization (here and in save data!)
    overflow = full_file['overflow'][:] / 1e0
    full_file.close()

    data = {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy,
        'overflow': overflow
    }

    return data

def save_data(data, data_file):
    """saves the given dict as h5py file. Has to satisfy the syntax of the original dataset"""
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']
    overflow = data['overflow']

    save_file = h5py.File(data_file, 'w')
    save_file.create_dataset('layer_0', data=layer_0*1e3)
    save_file.create_dataset('layer_1', data=layer_1*1e3)
    save_file.create_dataset('layer_2', data=layer_2*1e3)
    save_file.create_dataset('energy', data=energy*1e0)
    save_file.create_dataset('overflow', data=overflow*1e0)
    save_file.close()

def get_layer_sizes(data_flattened):

    if data_flattened.shape[1] >= 504:
        return 288, 144, 72
    elif data_flattened.shape[1] >= 288:
        return 72, 144, 72
    elif data_flattened.shape[1] >= 90:
        return 18, 36, 36
    elif data_flattened.shape[1] >= 45:
        return 18, 18, 9
    elif data_flattened.shape[1] >= 12:
        return 3, 6, 3
    
def get_layer_shapes(data_flattened):
    
    if data_flattened.shape[1] >= 504:
        return [3, 96], [12, 12], [12, 6]
    elif data_flattened.shape[1] >= 288:
        return [3, 24], [12, 12], [12, 6]
    elif data_flattened.shape[1] >= 90:
        return [3, 6], [6, 6], [6, 6]
    elif data_flattened.shape[1] >= 45:
        return [3, 6], [6, 3], [3, 3]
    elif data_flattened.shape[1] >= 12:
        return [1, 3], [6, 1], [3, 1]

def preprocess(data):
    """Preprocessing of the given data dict. Returns two numpy arrays containing the features and the conditions."""
        
    # Extract the arrays from the dict
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']
    
    # flatten the arrays
    layer0 = layer_0.reshape(layer_0.shape[0], -1)
    layer1 = layer_1.reshape(layer_1.shape[0], -1)
    layer2 = layer_2.reshape(layer_2.shape[0], -1)

    
    # If we use a single layer, make sure that the total energy is larger than the threshold
    x = np.concatenate((layer0, layer1, layer2), 1)

    # Mask to filter the events (e.g. ensure energy conservation)
    binary_mask = np.full(len(energy), True)
    # Ensure energy conservation
    binary_mask &= np.sum(x, axis=1) < energy[:,0]

    # Apply these two conditions
    x = x[binary_mask]
    c = energy[binary_mask]

    # adds the "energy dims", variables that represent the layer energies to the conditions.
    c = get_energy_dims(data=x, e_part=c)

    return x, c

def postprocess(data_flattened, conditions, threshold=1e-5, overflow=None):
    """Reverses the precprocessing and returns an dict that e.g. could be used for save_data."""
    
    # Input sanity checks
    assert len(data_flattened) == len(conditions)
    assert len(data_flattened.shape) == 2
    assert len(conditions.shape) == 2
    
    # Makes sure, that the original set is not modified inplace
    data_flattened = torch.clone(data_flattened)
    conditions = torch.clone(conditions)

    # Set all energies samller than a threshold to 0. Also prevents negative energies that might occur due to the alpha parameter in
    # the logit preprocessing
    data_flattened[data_flattened < threshold] = 0.

    # Reshape the layers to their original shape
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data_flattened)
    shape_layer_0, shape_layer_1, shape_layer_2 = get_layer_shapes(data_flattened=data_flattened)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1
    
    layer_0 = data_flattened[:, :l_0].reshape(-1, shape_layer_0[0], shape_layer_0[1]).cpu().numpy()
    layer_1 = data_flattened[:, l_0:l_01].reshape(-1, shape_layer_1[0], shape_layer_1[1]).cpu().numpy()
    layer_2 = data_flattened[:, l_01:].reshape(-1, shape_layer_2[0], shape_layer_2[1]).cpu().numpy()
    energy = conditions[:, [0]].cpu().numpy()
    
    # Adds an empty overflow to the dataset, if overflow is not specified
    if overflow is None:
        overflow = np.zeros((len(data_flattened), 3))
    else:
        overflow = overflow.cpu().numpy()

    return {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy,
        'overflow': overflow
    }

def get_energy_dims(data, e_part):
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1
    
    e0 = np.sum(data[..., :l_0], axis=1, keepdims=True)
    e1 = np.sum(data[..., l_0:l_01], axis=1, keepdims=True)
    e2 = np.sum(data[..., l_01:], axis=1, keepdims=True)
    # print(e0.min(), e1.min(), e2.min())
    u1 = (e0+e1+e2)/e_part
    u2 = e0/(e0+e1+e2)
    u3 = e1/(e1+e2+1e-7)
    
    return np.concatenate((e_part, u1, u2, u3), axis=1)
    
def normalize_layers(data_flattened, conditions):
    
    # Prevent inplace operations
    data_flattened = torch.clone(data_flattened)
    conditions = torch.clone(conditions)
    
    # Get the layer sizes
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data_flattened)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1    
    
    # Extract energy dimensions and incident energy from the conditions
    u1 = conditions[:, [1]]
    u2 = conditions[:, [2]]
    u3 = conditions[:, [3]]
    e_tot = u1*conditions[:, [0]]
    
    
    # Get layer energies
    e0 = u2*e_tot
    e1 = u3*(e_tot-e0)
    e2 = e_tot - e0 -e1
    
    # print(e0.min(), e1.min(), e2.min())

    # Normalize each layer by the layer energy
    data_flattened[..., :l_0] = data_flattened[..., :l_0] / e0
    data_flattened[..., l_0:l_01] = data_flattened[..., l_0:l_01] / (e1 + 1e-7)
    data_flattened[..., l_01:] = data_flattened[..., l_01:] / (e2 + 1e-7)
    
    return data_flattened

def unnormalize_layers(data_flattened, conditions):
    # Prevent inplace operations
    data_flattened = torch.clone(data_flattened)
    conditions = torch.clone(conditions)
    
    # Get the layer sizes
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data_flattened)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1 
    
    # Extract energy dimensions and incident energy from the conditions
    
    u1 = conditions[:, [1]]
    u2 = conditions[:, [2]]
    u3 = conditions[:, [3]]
    e_tot = u1*conditions[:, [0]]
    
    # Get layer energies
    e0 = u2*e_tot
    e1 = u3*(e_tot-e0)
    e2 = e_tot - e0 -e1
    
    # Normalize the layers to the correct energies
    data_flattened[..., :l_0]    = data_flattened[..., :l_0] / (torch.sum(data_flattened[..., :l_0], axis=1, keepdims=True) + 1e-7)
    data_flattened[..., l_0:l_01] = data_flattened[..., l_0:l_01] / (torch.sum(data_flattened[..., l_0:l_01], axis=1, keepdims=True) + 1e-7)
    data_flattened[..., l_01:]    = data_flattened[..., l_01:] / (torch.sum(data_flattened[..., l_01:], axis=1, keepdims=True) + 1e-7)
    data_flattened[..., :l_0]    = data_flattened[..., :l_0] * e0
    data_flattened[..., l_0:l_01] = data_flattened[..., l_0:l_01] * e1
    data_flattened[..., l_01:]    = data_flattened[..., l_01:] * e2
    
    return data_flattened

def get_loaders(data_file_train, data_file_test, batch_size, device='cpu', drop_last=False, shuffle=True):
    """Returns the dataloaders for the training of the CVAE"""
    
    # Create the train loader
    data_train, cond_train = preprocess(load_data(data_file_train))

    # Create the test loader
    data_test, cond_test = preprocess(load_data(data_file_test))

    # Convert the ndarrays into torch.tensors
    data_train = torch.tensor(data_train, device=device, dtype=torch.get_default_dtype())
    cond_train = torch.tensor(cond_train, device=device, dtype=torch.get_default_dtype())

    data_test = torch.tensor(data_test, device=device, dtype=torch.get_default_dtype())
    cond_test = torch.tensor(cond_test, device=device, dtype=torch.get_default_dtype())
    
    # Just to check that the function is not returning an error
    postprocess(data_train, cond_train)
    postprocess(data_test, cond_test)
    
    # Create the dataloaders
    loader_train = MyDataLoader(data_train, cond_train, batch_size, drop_last, shuffle)
    loader_test = MyDataLoader(data_test, cond_test, batch_size, drop_last, shuffle)
    return loader_train, loader_test