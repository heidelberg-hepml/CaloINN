import numpy as np
import h5py
import torch

from myDataLoader import MyDataLoader
from XMLHandler import XMLHandler
import HighLevelFeatures as HLF

def load_data_calo(filename, layer_boundaries):
    data = {}
    data_file = h5py.File(filename, 'r')
    data["energy"] = data_file["incident_energies"][:] / 1.e3
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end] / 1.e3
    data_file.close()
    
    return data

def load_data(filename, particle_type,  xml_filename, threshold=1e-5):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, 
    filename=xml_filename)
    
    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    data_file = h5py.File(filename, 'r')
    data["energy"] = data_file["incident_energies"][:] / 1.e3
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end] / 1.e3
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
    incident_energies *= 1.e3
    
    # concatenate the layers and renormalize them, too           
    showers = np.concatenate(layers, axis=1) * 1.e3
            
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
    
    #add_noise = np.random.rand(*x.shape)*1.0e-6
    #x += add_noise

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
    #all_conditions = [c] + extra_dims
    #c = np.concatenate(all_conditions, axis=1)
    extra_dims = np.concatenate(extra_dims, axis=1)

    #extra_dims[:,0] /= 3.5

    return c, extra_dims

def preprocess(data, layer_boundaries, eps=1.e-10):
    """Transforms the dict 'data' into the ndarray 'x'. Furthermore, the events
    are masked and the extra dims are appended to the incident energies"""
    energy, layers = get_energy_and_sorted_layers(data)

    # Concatenate the layers
    x = np.concatenate(layers, axis=1)

    # Remove all no-interaction events (only 0.7%)
    #binary_mask = np.sum(x, axis=1) > 0
    #binary_mask &= np.sum(x, axis=1) < energy[:,0]
    
    #x = x[binary_mask]
    #c = energy[binary_mask]
    c = energy
    c, extra_dims = get_energy_dims(x, c, layer_boundaries, eps)

    #binary_mask &= extra_dims[:,0] < 3.0
    
    #x = x[binary_mask]
    #c = energy[binary_mask]
    #extra_dims = extra_dims[binary_mask]
    #extra_dims[:, 0] /= extra_dims[:,0].max()

    x = normalize_layers(x, c, layer_boundaries)
    x = np.concatenate((x, extra_dims), axis=1)
    
    return x, c

def postprocess(x, c, layer_boundaries, quantiles, threshold=1e-4):
    """Reverses the effect of the preprocess funtion"""
    
    # Input sanity checks
    assert len(x) == len(c)
    assert len(x.shape) == 2
    assert len(x.shape) == 2
    
    # Makes sure, that the original set is not modified inplace
    x = np.copy(x)
    c = np.copy(c)

    # Set all energies smaller than a threshold to 0. Also prevents negative energies that might occur due to the alpha parameter in
    # the logit preprocessing
    # TODO: Pipe to params
    #x[x < threshold] = 0.
    x[x < quantiles] = 0.0
    x = unnormalize_layers(x, c, layer_boundaries)
    print(x.min())
    #x[x < 0] = 0
    #x[x < threshold] = 0

    # Create a new dict 'data' for the output
    data = {}
    data["energy"] = c[..., [0]]
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = x[..., layer_start:layer_end]

    return data

def normalize_layers(x, c, layer_boundaries, eps=1.e-10):
    """Normalizes each layer by its energy"""
    
    # Prevent inplace operations
    x = np.copy(x)
    c = np.copy(c)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = c[..., 1:number_of_layers+1]
    
    # Use the exact layer energies for numerical stability
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        x[..., layer_start:layer_end] = x[..., layer_start:layer_end] / ( np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
        
    return x

def unnormalize_layers(x, c, layer_boundaries, eps=1.e-10):
    """Reverses the effect of the normalize_layers function"""
    
    # Here we should not use clone, since it might result
    # in a memory leak, if this functions is used on tensors
    # with gradients. Instead, we use a different output tensor
    # to prevent inplace operations.
    output = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64)
    
    # Get the number of layers
    number_of_layers = len(layer_boundaries) - 1

    # Split up the conditions
    incident_energy = c[..., [0]]
    extra_dims = x[..., -number_of_layers:]
    #extra_dims[:,0] *= 3.5
    extra_dims[:, (-number_of_layers+1):] = np.clip(extra_dims[:, (-number_of_layers+1):], a_min=0., a_max=1.)   #clipping 
    x = x[:, :-number_of_layers]
    
    layer_energies = []
    en_tot = np.multiply(incident_energy.flatten(), extra_dims[:,0])
    cum_sum = np.zeros_like(en_tot, dtype=np.float64)
    for i in range(extra_dims.shape[-1]-1):
        ens = (en_tot - cum_sum)*extra_dims[:,i+1]
        layer_energies.append(ens)
        cum_sum += ens

    layer_energies.append((en_tot - cum_sum))
    layer_energies = np.vstack(layer_energies).T
    # Normalize each layer and multiply it with its original energy
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        output[..., layer_start:layer_end] = x[..., layer_start:layer_end] * layer_energies[..., [layer_index]]  / \
                                             (np.sum(x[..., layer_start:layer_end], axis=1, keepdims=True) + eps)
    print(output.min(), output.max())
    return output

def save_hlf(hlf, filename):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(hlf, file)
    print("Saving file with high-level features DONE.")

def get_loaders(filename, xml_filename, particle_type, val_frac, batch_size, eps=1.e-10, device='cpu', drop_last=False, shuffle=True, width_noise=0.0):
    """Creates the dataloaders used to train the VAE model."""
    
    # load the data from the hdf5 file
    data, layer_boundaries = load_data(filename, particle_type, xml_filename=xml_filename)

    # preprocess the data and append the extra dims
    x, c = preprocess(data, layer_boundaries, eps)
    
    # Create an index array, used for splitting into train and val set
    number_of_samples = len(x)
    if shuffle:
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
    #_ = postprocess(x_trn, c_trn, layer_boundaries)
    #_ = postprocess(x_val, c_val, layer_boundaries)
    
    # Create the dataloaders
    trn_loader = MyDataLoader(x_trn, c_trn, batch_size, drop_last, shuffle, width_noise)
    val_loader = MyDataLoader(x_val, c_val, batch_size, drop_last, shuffle, width_noise)
    return trn_loader, val_loader, layer_boundaries

def get_hlf(shower, particle_type, layer_boundaries, threshold=1.e-4):
    "returns a hlf class needed for plotting"
    #x = x.cpu()
    #c = c.cpu()
    
    hlf = HLF.HighLevelFeatures(particle_type,
                                f"/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_1_{particle_type}s.xml")
    
    # like in save function:
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(shower)
    
    # renormalize the energies
    #incident_energies = shower['incident_energies']
    incident_energies *= 1.e3
    
    # concatenate the layers and renormalize them, too 
    #showers = shower['showers']
    showers = np.concatenate(layers, axis=1) * 1.e3
    
    
    hlf.CalculateFeatures(showers, threshold * 1.e3)
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

