import numpy as np
import h5py
import torch
from copy import deepcopy
from myDataLoader import MyDataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import pickle

import os
import warnings




def normalize_layers(x, layer_boundaries, c=None, eps=1.e-10):
    """Normalizes each layer by its energy"""
    
    # Prevent inplace operations
    output = torch.zeros_like(x).to(x.device)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Use the passed c to speed up computation. Used if no noise is added    
    if c is not None:
        # Use the exact layer energies for numerical stability  
        layer_energies = c[..., -number_of_layers:]
        
        for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
            output[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (layer_energies[..., [layer_index]] + eps)
    
    # We have to recompute the layer normalization 
    else:
        for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
            output[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (torch.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
        
    return output

def unnormalize_layers(x, c, layer_boundaries, eps=1.e-10, noise_width=None):
    """Reverses the effect of the normalize_layers function"""
    
    # Here we should not use clone, since it might result
    # in a memory leak, if this functions is used on tensors
    # with gradients. Instead, we use a different output tensor
    # to prevent inplace operations.
    output = torch.zeros_like(x).to(x.device)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = c[..., 1:number_of_layers+1]
    layer_energies = c[..., -number_of_layers:]
    # brightest_voxels = c[..., number_of_layers+1:(2*number_of_layers)+1]

    
    
    # Normalize each layer and multiply it with its original energy
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        
        if noise_width is not None:
            number_of_voxels = (layer_end - layer_start)
            noise_correction = noise_width / 2 * number_of_voxels
        else:
            noise_correction = 0
            noise_width = 0
        
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * (layer_energies[..., [layer_index]] + noise_correction) / \
            (torch.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)

    return output


def load_data(filename, used_layers=None, ):
    """Loads the data for the ML training from an hdf5 file"""
    
    # layer_boundaries = [0, size_layer_0, size_layer_0+size_layer_1, ...]
    # data = {"incident_energy": energy, "energy_layer_0": layer_0, ...}
    # filename = "/afs/cern.ch/user/f/fernst/FrozenShowerInputSamples/eta_020_binned/dataset_eta_020.hdf5"
    
    file = h5py.File(filename, 'r')
    
    if used_layers is None:
        used_layers = [int(key.split("_")[-1]) for key in file.keys() if "bin" not in key and "layer" in key]
        
    used_layers = torch.tensor(used_layers, dtype=torch.int32)    
    used_layers = torch.sort(used_layers)[0]
        
    print(f"Using layers {used_layers}")
        
    # Store the layer energies
    layers = []
    for layer_index in used_layers:
        layers.append(torch.tensor(file[f"energy_layer_{layer_index}"][:], dtype=torch.get_default_dtype()))
    
    # Store the incident energy
    energy = torch.tensor(file["incident_energy"][:], dtype=torch.get_default_dtype())[:, None]
    
    # Create the layer boundaries list
    layer_boundaries = [0]
    for layer in layers:
        layer_boundaries.append(layer.shape[1] + layer_boundaries[-1])
        
    coordinates = [[],[]]
    
    for layer_index in used_layers:
        alpha = torch.tensor(file[f"binsize_alpha_layer_{layer_index}"][:], dtype=torch.get_default_dtype())/2 \
            +   torch.tensor(file[f"binstart_alpha_layer_{layer_index}"][:], dtype=torch.get_default_dtype())
            
        radius = torch.tensor(file[f"binsize_radius_layer_{layer_index}"][:], dtype=torch.get_default_dtype())/2 \
            +    torch.tensor(file[f"binstart_radius_layer_{layer_index}"][:], dtype=torch.get_default_dtype())
        
        eta = radius * torch.cos(alpha)
        phi = radius * torch.sin(alpha)
        
        coordinates[0].append(eta)
        coordinates[1].append(phi)
        
    coordinates[0] = torch.cat(coordinates[0])
    coordinates[1] = torch.cat(coordinates[1])
    
    coordinates = torch.stack(coordinates)
    
    # Concatenate the layers
    x = torch.cat(layers, axis=1)
    
    # Turn x into MeV scale
    x *= energy    

    file.close()
    
    return x, energy, layer_boundaries, coordinates

def separate_negative_energies(x, layer_boundaries):
    negative_layers = []
        
    new_layers = [x]
    for i, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        layer = x[..., layer_start:layer_end]
        if torch.any(layer < 0):
            
            negative_layers.append(i)
            
            new_layers.append(-torch.clip(layer, None, 0))
            
            x[..., layer_start:layer_end] = torch.clip(layer, 0, None)
            
            layer_boundaries.append(layer_boundaries[-1] + layer.shape[1])
            
    print(f"Fixed {len(negative_layers)} negative layers")
    
    x = torch.cat(new_layers, axis=1)
    
    return x, layer_boundaries, negative_layers
    
def get_energy_dims(x, c, layer_boundaries, eps=1.e-10):
    """Appends the extra dimensions and the layer energies to the conditions
    The layer energies will always be the last #layers entries, the extra dims will
    be the #layers entries directly after the first entry - the incident energy.
    Inbetween additional features might be appended as further conditions"""
    
    x = torch.clone(x)
    c = torch.clone(c)

    layer_energies = []

    for layer_start, layer_end in zip(layer_boundaries[:-1], layer_boundaries[1:]):
        
        # Compute total energy of current layer
        layer_energy = torch.sum(x[..., layer_start:layer_end], axis=1, keepdims=True)
        
        # Store its energy for later
        layer_energies.append(layer_energy)
        
        
    layer_energies_torch = torch.cat(layer_energies, axis=1)
        
    # Compute the generalized extra dimensions
    extra_dims = [torch.sum(layer_energies_torch, axis=1, keepdims=True) / c]

    for layer_index in range(len(layer_boundaries)-2):
        extra_dim = layer_energies_torch[..., [layer_index]] / (torch.sum(layer_energies_torch[..., layer_index:], axis=1, keepdims=True) + eps)
        extra_dims.append(extra_dim)
        
    # Collect all the conditions
    all_conditions = [c] + extra_dims + layer_energies
    c = torch.cat(all_conditions, axis=1)
    
    return c
            
def preprocess(x, energy, layer_boundaries, eps=1.e-10):
    """Transforms the list 'layers' into the ndarray 'x'. Furthermore, the events
    are masked and the extra dims are appended to the incident energies"""
        
    x, layer_boundaries, negative_layers = separate_negative_energies(x, layer_boundaries)

    binary_mask = torch.full((len(energy),), True)

    # Rescale the energies by an arbitrary factor of 2 -> Only loose O(10) showers instead of ~50%
    # Has to be reversed in the postprocess loop
    x = x/2
    
    # Ensure energy conservation
    binary_mask &= torch.sum(x, axis=1) < energy[:, 0]
    
    # Remove all no-interaction events (= 4 Events in the dataset)
    binary_mask &= torch.sum(x, axis=1) > 0
    
    print(f"Removed {len(energy) - torch.sum(binary_mask)} of {len(energy)} events ({100*(1-torch.sum(binary_mask)/len(energy)):.2f}%)")

    x = x[binary_mask]
    c = energy[binary_mask]

    c = get_energy_dims(x, c, layer_boundaries, eps)
    
    return x, c, negative_layers

def recombine_negative_energies(x, layer_boundaries, negative_layers):
    """Subtracts the negative layers from the positive layers to receive the original data."""
    
    n_layers = len(negative_layers)
    
    for i, layer_index in enumerate(negative_layers):
        positive_layer = x[..., layer_boundaries[layer_index]:layer_boundaries[layer_index+1]]
        negative_layer = x[..., layer_boundaries[-n_layers+i-1]:layer_boundaries[-n_layers+i]]
        
        x[..., layer_boundaries[layer_index]:layer_boundaries[layer_index+1]] = positive_layer - negative_layer
    
    return x[..., :layer_boundaries[-n_layers-1]], layer_boundaries[:-n_layers]

def postprocess(x, c, layer_boundaries, negative_layers, threshold=1e-10, inplace=False):
    """Reverses the effect of the preprocess funtion"""
    
    # Input sanity checks
    assert len(x) == len(c)
    assert len(c.shape) == 2
    assert len(x.shape) == 2
    
    if not inplace:
        # Makes sure, that the original set is not modified inplace
        x = torch.clone(x)
        c = torch.clone(c)
           
    # Set all energies smaller than a threshold to 0. Also prevents negative energies that might occur due to the alpha parameter in
    # the logit preprocessing
    x[x < threshold] = 0.
    
    # Reverse the rescaling that was used before
    x = x*2
    
    x, layer_boundaries = recombine_negative_energies(x, layer_boundaries, negative_layers)
    

    return x, c[..., [0]], layer_boundaries

def get_loaders(filename, val_frac, batch_size, used_layers=None, eps=1.e-10, device='cpu', drop_last=False, shuffle=True, save_memory=False, width_noise=0):
    """Creates the dataloaders used to train the VAE model."""
    
    # load the data from the hdf5 file
    x, energy, layer_boundaries, coordinates = load_data(filename, used_layers=used_layers)

    # preprocess the data and append the extra dims
    x, c, negative_layers = preprocess(x, energy, layer_boundaries, eps)
    
    # Create an index array, used for splitting into train and val set
    number_of_samples = len(x)
    
    # Dont want to mix train and test set, when loading -> No random permutation
    full_index = np.arange(number_of_samples)

    # Split the data
    number_of_val_samples = int(number_of_samples * val_frac)
    number_of_trn_samples = number_of_samples - number_of_val_samples

    trn_index = full_index[:number_of_trn_samples]
    val_index = full_index[number_of_trn_samples:]
    
    if save_memory:
        device = 'cpu'
    
    x_trn = x[trn_index].to(device)
    c_trn = c[trn_index].to(device)
    
    x_val = x[val_index].to(device)
    c_val = c[val_index].to(device)

    # # Call the postprocess func to make sure that it runs through
    # postprocess(x_trn, c_trn, layer_boundaries)
    # postprocess(x_val, c_val, layer_boundaries)
    
    # Create the dataloaders
    trn_loader = MyDataLoader(x_trn, c_trn, batch_size, drop_last, shuffle, width_noise=width_noise)
    val_loader = MyDataLoader(x_val, c_val, batch_size, drop_last, shuffle, width_noise=width_noise)
    return trn_loader, val_loader, layer_boundaries, negative_layers, coordinates
            