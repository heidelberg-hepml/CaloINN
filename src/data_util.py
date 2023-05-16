import numpy as np
import h5py
import torch
from copy import deepcopy
from myDataLoader import MyDataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from XMLHandler import XMLHandler
import HighLevelFeatures as HLF
import pickle

import os
import warnings


def load_data(filename, particle_type, dataset=1):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    if dataset==1:
        xml_handler = XMLHandler(particle_name=particle_type, 
        filename=f'/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_1_{particle_type}s.xml')
    else:
        xml_handler = XMLHandler(particle_name=particle_type, 
        filename=f'/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_{dataset}.xml')
    
    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    data_file = h5py.File(filename, 'r')
    data["energy"] = data_file["incident_energies"][:] / 1.e5
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end] / 1.e5
    data_file.close()
    
    return data, layer_boundaries

def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""
    
    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data)-1
    
    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"
        
        layers.append(data[layer])
        
            
    return energy, layers

def save_data(data, filename):
    """Saves the data with the same format as dataset 1 from the calo challenge"""
    
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(data)
    
    # renormalize the energies
    incident_energies *= 1.e5
    
    # concatenate the layers and renormalize them, too           
    showers = np.concatenate(layers, axis=1) * 1.e5
            
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('incident_energies', data=incident_energies)
    save_file.create_dataset('showers', data=showers)
    save_file.close()            
  
def get_energy_dims(x, c, layer_boundaries, eps=1.e-10):
    """Appends the extra dimensions and the layer energies to the conditions
    The layer energies will always be the last #layers entries, the extra dims will
    be the #layers entries directly after the first entry - the incident energy.
    Inbetween additional features might be appended as further conditions"""
    
    x = np.copy(x)
    c = np.copy(c)

    layer_energies = []

    for layer_start, layer_end in zip(layer_boundaries[:-1], layer_boundaries[1:]):
        
        # Compute total energy of current layer
        layer_energy = np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True)
        
        # Normalize current layer
        x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (layer_energy + eps)
        
        # Store its energy for later
        layer_energies.append(layer_energy)
        
    layer_energies_np = np.array(layer_energies).T[0]
        
    # Compute the generalized extra dimensions
    extra_dims = [np.sum(layer_energies_np, axis=1, keepdims=True) / c]

    for layer_index in range(len(layer_boundaries)-2):
        extra_dim = layer_energies_np[..., [layer_index]] / (np.sum(layer_energies_np[..., layer_index:], axis=1, keepdims=True) + eps)
        extra_dims.append(extra_dim)
        
    # Collect all the conditions
    all_conditions = [c] + extra_dims + layer_energies
    c = np.concatenate(all_conditions, axis=1)
    
    return c

def preprocess(data, layer_boundaries, eps=1.e-10):
    """Transforms the dict 'data' into the ndarray 'x'. Furthermore, the events
    are masked and the extra dims are appended to the incident energies"""
    energy, layers = get_energy_and_sorted_layers(data)

    # Concatenate the layers
    x = np.concatenate(layers, axis=1)

    binary_mask = np.full(len(energy), True)

    # TODO: We loose about 20% of our events this way. Might reconsider the masking here...
    # EDIT: Rescale the energies by an arbitrary factor of 2 -> Only loose 10 showers instead of ~20 000
    # Has to be reversed in the postprocess loop
    x = x/2
    
    
    # Ensure energy conservation
    binary_mask &= np.sum(x, axis=1) < energy[:,0]
    # Remove all no-interaction events (only 0.7%)
    binary_mask &= np.sum(x, axis=1) > 0

    x = x[binary_mask]
    c = energy[binary_mask]

    c = get_energy_dims(x, c, layer_boundaries, eps)
    
    return x, c

def postprocess(x, c, layer_boundaries, threshold=1e-4):
    """Reverses the effect of the preprocess funtion"""
    
    # Input sanity checks
    assert len(x) == len(c)
    assert len(x.shape) == 2
    assert len(x.shape) == 2
    
    # Makes sure, that the original set is not modified inplace
    x = torch.clone(x)
    c = torch.clone(c)

    # Set all energies smaller than a threshold to 0. Also prevents negative energies that might occur due to the alpha parameter in
    # the logit preprocessing
    x[x < threshold] = 0.
    
    # Reverse the rescaling that was used before
    x = x*2
    
    # Create a new dict 'data' for the output
    data = {}
    data["energy"] = c[..., [0]].cpu().numpy()
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = x[..., layer_start:layer_end].cpu().numpy()

    return data

def normalize_layers(x, c, layer_boundaries, eps=1.e-10):
    """Normalizes each layer by its energy"""
    
    # Prevent inplace operations
    x = torch.clone(x)
    c = torch.clone(c)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = c[..., 1:number_of_layers+1]
    layer_energies = c[..., -number_of_layers:]
    
    # Use the exact layer energies for numerical stability
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / (layer_energies[..., [layer_index]] + eps)
        
    return x

def unnormalize_layers(x, c, layer_boundaries, eps=1.e-10):
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
    
    # Normalize each layer and multiply it with its original energy
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * layer_energies[..., [layer_index]] / \
                                             (torch.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)

    return output

def get_hlf(x, c, particle_type, layer_boundaries, threshold=1.e-4, dataset=1):
    "returns a hlf class needed for plotting"
    x = x.cpu()
    c = c.cpu()
    
    if dataset == 1:
        hlf = HLF.HighLevelFeatures(particle_type,
                                    f"/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_1_{particle_type}s.xml")
    else:
        hlf = HLF.HighLevelFeatures(particle_type,
                                    f"/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_{dataset}.xml")
    
    # Maybe we will do more than just thresholding in postprocess someday. So, we should call it here as well.
    data = postprocess(x, c, layer_boundaries, threshold)
    
    # like in save function:
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(data)
    
    # renormalize the energies
    incident_energies *= 1.e5
    
    # concatenate the layers and renormalize them, too           
    showers = np.concatenate(layers, axis=1) * 1.e5
    
    
    hlf.CalculateFeatures(showers, threshold * 1.e5)
    hlf.Einc = incident_energies
    hlf.showers = showers
    
    return hlf

def save_hlf(hlf, filename):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(hlf, file)
    print("Saving file with high-level features DONE.")

def get_loaders(filename, particle_type, val_frac, batch_size, eps=1.e-10, device='cpu', drop_last=False, shuffle=False, dataset=1,
                e_inc_index=None):
    """Creates the dataloaders used to train the VAE model."""
    
    # load the data from the hdf5 file
    data, layer_boundaries = load_data(filename, particle_type, dataset=dataset)

    # preprocess the data and append the extra dims
    x, c = preprocess(data, layer_boundaries, eps)
    
    if e_inc_index is not None:
        assert type(e_inc_index) == int
        e_incs = np.unique(c[..., 0])
        mask = (c[..., 0] == e_incs[e_inc_index])
        print(f"Use incident energy {e_incs[e_inc_index]} ({mask.sum()} events)")
        
        print(x.shape, c.shape)
        x = x[mask]
        c = c[mask]
        
        print(x.shape, c.shape)
        
        
    
    # Create an index array, used for splitting into train and val set
    number_of_samples = len(x)
    
    # Dont want to mix train and test set, when loading!
    if False:
        full_index = np.random.choice(number_of_samples, number_of_samples, replace=False)
    else:
        full_index = np.arange(number_of_samples)

    # Split the data
    number_of_val_samples = int(number_of_samples * val_frac)
    number_of_trn_samples = number_of_samples - number_of_val_samples

    trn_index = full_index[:number_of_trn_samples]
    val_index = full_index[number_of_trn_samples:]
    
    x_trn = x[trn_index]
    c_trn = c[trn_index]
    
    x_val = x[val_index]
    c_val = c[val_index]  
    
    # Cast into torch tensors
    x_trn = torch.tensor(x_trn, device=device, dtype=torch.get_default_dtype())
    c_trn = torch.tensor(c_trn, device=device, dtype=torch.get_default_dtype())
    x_val = torch.tensor(x_val, device=device, dtype=torch.get_default_dtype())
    c_val = torch.tensor(c_val, device=device, dtype=torch.get_default_dtype())

    # Call the postprocess func to make sure that it runs through
    _ = postprocess(x_trn, c_trn, layer_boundaries)
    _ = postprocess(x_val, c_val, layer_boundaries)
    
    # Create the dataloaders
    trn_loader = MyDataLoader(x_trn, c_trn, batch_size, drop_last, shuffle)
    val_loader = MyDataLoader(x_val, c_val, batch_size, drop_last, shuffle)
    return trn_loader, val_loader, layer_boundaries
