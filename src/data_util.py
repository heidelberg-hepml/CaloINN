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


def load_data(filename, particle_type):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, 
    filename=f'/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_1_{particle_type}s.xml')
    
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

    # TODO: We loose about 5% of our events this way. Might reconsider the masking here...
    # Ensure energy conservation
    binary_mask &= np.sum(x, axis=1) < energy[:,0]
    # Remove all no-interaction events (only 0.7%)
    binary_mask &= np.sum(x, axis=1) > 0

    x = x[binary_mask]
    c = energy[binary_mask]

    c = get_energy_dims(x, c, layer_boundaries, eps)
    
    return x, c

def postprocess(x, c, layer_boundaries, threshold=1e-10):
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
    # TODO: Pipe to params
    x[x < threshold] = 0.
    
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

def get_hlf(x, c, particle_type, layer_boundaries, threshold=1.e-10):
    "returns a hlf class needed for plotting"
    x = x.cpu()
    c = c.cpu()
    
    hlf = HLF.HighLevelFeatures(particle_type,
                                f"/remote/gpu06/ernst/Master_Thesis/vae_calo_challenge/CaloINN/calo_challenge/code/binning_dataset_1_{particle_type}s.xml")
    
    # Maybe we will do more than just thresholding in postprocess someday. So, we should call it here as well.
    data = postprocess(x, c, layer_boundaries, threshold)
    
    # like in save function:
    # extract the needed data
    incident_energies, layers = get_energy_and_sorted_layers(data)
    
    # renormalize the energies
    incident_energies *= 1.e5
    
    # concatenate the layers and renormalize them, too           
    showers = np.concatenate(layers, axis=1) * 1.e5
    
    
    hlf.CalculateFeatures(showers)
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

def get_loaders(filename, particle_type, val_frac, batch_size, eps=1.e-10, device='cpu', drop_last=False, shuffle=True):
    """Creates the dataloaders used to train the VAE model."""
    
    # load the data from the hdf5 file
    data, layer_boundaries = load_data(filename, particle_type)

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
    _ = postprocess(x_trn, c_trn, layer_boundaries)
    _ = postprocess(x_val, c_val, layer_boundaries)
    
    # Create the dataloaders
    trn_loader = MyDataLoader(x_trn, c_trn, batch_size, drop_last, shuffle)
    val_loader = MyDataLoader(x_val, c_val, batch_size, drop_last, shuffle)
    return trn_loader, val_loader, layer_boundaries


# Dataset for the classifier test:
class CaloDataset(Dataset):
    """CaloGAN dataset, updated to handle a variable amount of layers"""

    def __init__(self, data_file, transform=None,apply_logit=True,
                 with_noise=False, noise_width=1e-8, return_label=True):
        """CaloGAN dataset, updated to handel a variable number of calorimeter layers
        Args:
            data_file (dict): Underlying data file for the dataset. Should have the form as produced by merge dataset
            transform (str or dict, optional): "L_norm" = normalize the data to the layer energy or "E_norm" = normalize to the total energy.
                If a string is passed, it is used for all layers, in case of a dict the keys of the dict must match with the keys of the layers.
                Then the transformations can be specified for each layer seperately.
                Defaults to None.
            apply_logit (bool, optional): Should a logit transformation be applied. Defaults to True.
            with_noise (bool, optional): Should noise be applied to the data. Defaults to False.
            noise_width (_type_, optional): How large should the applied noise be
                No effect if with_noise=False. Defaults to 1e-8.
            return_label (bool, optional): Whether the labels for the training should be returned. Defaults to True. (Sample == 0, true == 1)
        """
        

        self.full_file = dict(data_file)

        self.apply_logit = apply_logit
        self.with_noise = with_noise
        self.noise_width = noise_width

        self.return_label = return_label
        
        self.input_dims = {}
        self.input_size = {}
        
        # We need the energy information
        assert "energy" in data_file, "The key 'energy' is missing in the given data file"
        
        # We need labels to train the classifier
        assert "label" in data_file, "The key 'label' is missing in the given data file"
        
        # We need the overflow information
        assert "overflow" in data_file, "The key 'overflow' is missing in the given data file"
        
        # Save the labels of the calorimeter layers
        self.layer_names = list(data_file.keys())
        self.layer_names.remove("energy")
        self.layer_names.remove("overflow")
        self.layer_names.remove("label")
        
        # At least one additional layer for training should be there
        assert len(self.layer_names) > 0, "No calorimeter layer found in the given data file"
        
        # The data should have the format (datapoints, x, y)
        for layer in self.layer_names:
            assert len(self.full_file[layer].shape) == 3, f"Datashape of the calorimeter layer {layer} does not match"
        
        # Prepare the layer transformations
        if transform is not None:
            if type(transform) == str:
                self.transform = {}
                for layer in self.layer_names:
                    self.transform["layer"] = transform
            else:
                assert self.layer_names in transform, f"The keys for the transformation dictionary do not match with {self.layer_names}"
                self.transform = transform
        else:
            self.transform = transform
        
        for key in data_file:
            if key == "energy":
                # Normalize to 100 GeV
                self.full_file[key][:] = self.full_file[key][:] / 1e2
                
            elif key == "overflow":
                # Normalize to 100 GeV
                self.full_file[key][:] = self.full_file[key][:] / 1e5
                
            elif key != "label":
                # Normalize to 100 GeV
                self.full_file[key][:] = self.full_file[key][:] / 1e5
                
                # Save the shape of the input layers
                self.input_dims[key] = self.full_file[key][0].shape
                self.input_size[key] = self.full_file[key][0].flatten().shape
                
                # Save the amount of datapoints in the dataset
                self.size = len(self.full_file[key])        

    def __len__(self):
        # assuming file was written correctly
        #return len(self.full_file['energy'])
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        sample = dict(self.full_file)
        
        for layer in self.layer_names:
            without_noise = self.full_file[layer][idx]
            
            # Apply noise to the data
            if self.with_noise:
                # The data without the noise
                # Add the noise and save the result
                layer_data = self.__add_noise(without_noise)
            else:
                layer_data = without_noise
                
            # save the total energies per layer
            layer_energies = layer_data.sum(axis=(-1,-2), keepdims=True)
            
            # Apply the given transformation if needed
            if self.transform is not None:
                transform = self.transform[layer]
                if transform == "E_norm":
                    layer_data = layer_data / self.full_file["energy"]
                elif transform == "L_norm":
                    layer_data = layer_data / layer_energies
                else:
                    raise KeyError(f"The transformation {transform} is not supported")
           
           # Apply a logit transformation if requested
            if self.apply_logit:
                layer_data = self.__logit_trafo(layer_data)
            
            # Update the samples file
            sample[layer] = layer_data
            
            # Save the layer energies
            # Do I really need them? -> Yes they are used!
            sample[layer + "_E"] = layer_energies.squeeze()
                    

        if not self.return_label:
            #TODO: Maybe make the pass of "label" optional. Edit here if doing so !!!
            sample.pop("label", None)
        else:
            sample["label"] = sample["label"][idx]
            
        
        sample["energy"] = sample["energy"][idx]
        sample["overflow"] = sample["overflow"][idx]


        return sample
    
    def __add_noise(self, input_tensor):
        """Adds noise to the given tensor according to self.noise_width"""
        
        noise = np.random.rand(*input_tensor.shape)*self.noise_width
        return input_tensor+noise

    def __logit_trafo(self, x):
        # TODO: Maybe allow modifications of ALPHA?
        
        ALPHA = 1e-6
        def logit(x):
            return np.log(x / (1.0 - x))
        
        local_x = ALPHA + (1. - 2.*ALPHA) * x
        return logit(local_x)

# Functions for the classifier test:
def h5py_to_dict(file, dtype=torch.get_default_dtype()):
    """Converts a given h5py file into a dictionary.
    Args:
        file (h5py file): h5py file that is converted
        dtype (torch.dtype, optional): The dtype for the tensors in the dict. Defaults to torch.get_default_dtype().
    Returns:
        dict: converted input
    """
    # TODO: watch out for dtype bug!
    new_file = {}
    for key in file:
        new_file[key] = torch.tensor(np.array(file[key]), dtype=dtype)
    return new_file

def merge_datasets(sample, original):
    """Merge the two given files in order to 
    perform the classifier test with them later.
    Args:
        sample (dict): dataset sampled by the network
        original (dict): original ground truth dataset
    Returns:
        dict: dataset received by merging the two given sets
    """
    # Merge the two datasets
    key_len_1 = []
    for key in sample:
        key_len_1.append(len(sample[key]))
    key_len_1 = np.array(key_len_1)
    
    key_len_2 = []
    for key in original.keys():
        key_len_2.append(len(original[key]))
    key_len_2 = np.array(key_len_2)
    
    # Make sure, that all entries in the dicts have the same length
    assert np.all(key_len_1 == key_len_1[0]), f"{key_len_1}, {key_len_1[0]}"
    assert np.all(key_len_2 == key_len_2[0]), f"{key_len_2}, {key_len_2[0]}"
    
    # Make sure that the dicts are equal sized
    assert np.all(key_len_1 == key_len_2), f"{key_len_1}, {key_len_2}"
    assert len(sample) == len(original)
    
    # Shuffle the elements of the merged dataset
    shuffle_order = np.arange(key_len_1[0]+key_len_2[0])
    np.random.shuffle(shuffle_order)

    merged_file = {}
    # Merge the datasets
    for key in original:
        data1 = sample[key][:]
        data2 = original[key][:]
        data = np.concatenate([data1, data2])
        merged_file[key] = data[shuffle_order]

    # Create ground truth lables for the classifier test
    truth1 = np.zeros(key_len_1[0])
    truth2 = np.ones(key_len_2[0])
    truth = np.concatenate([truth1, truth2])
    merged_file['label'] = truth[shuffle_order]
    
    return merged_file

def split_dataset(input_dataset, val_fraction=0.2, test_fraction=0.2, save_dir=None, file_name=None):
    """Splits a given dataset into train, test and validation sets according to the given split parameters.
    Args:
        input_dataset (dict): Dataset that is splitted
        val_fraction (float, optional): Fraction of the validation part. Defaults to 0.2.
        test_fraction (float, optional): Fraction of the test part. Defaults to 0.2.
        save_dir (str, optional): Files are saved if a path is given SLOW! Defaults to None.
        file_name (str, optional): Only used if save_dir is passed. Will be used as filename_suffix. Defaults to None.
    Returns:
        dict: the three datasets in the order (training set, validation set, test set)
    """
    key_len = []
    for key in input_dataset:
        key_len.append(len(input_dataset[key]))
    key_len = np.array(key_len)

    # Make sure, that all entries in the dicts have the same length
    assert np.all(key_len==key_len[0])

    # Compute the splitting points
    cut_index_val = int(val_fraction * key_len[0])
    cut_index_test = cut_index_val + int(test_fraction * key_len[0])

    val_dataset = {}
    test_dataset = {}
    train_dataset = {}
    
    for key in input_dataset:
        input_dataset[key] = torch.tensor(input_dataset[key], dtype=torch.get_default_dtype())
        val_dataset[key] = input_dataset[key][:cut_index_val]
        test_dataset[key] = input_dataset[key][cut_index_val:cut_index_test]
        train_dataset[key] = input_dataset[key][cut_index_test:]
    
    # Can also save the three files.
    # Is rather slow, however.
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
        if file_name is None:
            file_name = "samples.h5py"
            
        train_path = os.path.join(save_dir, "train_"+file_name)
        val_path = os.path.join(save_dir, "validation_"+file_name)
        test_path = os.path.join(save_dir, "test_"+file_name)
        
        train_file = h5py.File(train_path, 'w')
        val_file = h5py.File(val_path, 'w')
        test_file = h5py.File(test_path, 'w')

        for key in input_dataset:
            val_file.create_dataset(key, data=input_dataset[key][:cut_index_val])
            test_file.create_dataset(key, data=input_dataset[key][cut_index_val:cut_index_test])
            train_file.create_dataset(key, data=input_dataset[key][cut_index_test:])
        
    return train_dataset, val_dataset, test_dataset

def get_classifier_loaders(test_trainer, params, doc, device, drop_layers=None, postprocessing=False, val_fraction = 0.2, test_fraction = 0.2):
    """Returns the dataloaders for the classifier test sampling from the passed model
    Args:
        params (dict): params dictionary from the training (origin: yaml file)
        doc (documeter): An instance of the documenter class responsible for documenting the run
        device (str): device used for pytorch calculations
        trainer: trainer with a model to be sampled fron. Must have a trainer.generate function like the INN_trainer.
        drop_layers (list of strings, optional): A container wrapping the names of the layers that should be dropped.
            Defaults to None.
    Returns:
        torch.utils.data.DataLoader: Training dataloader
        torch.utils.data.DataLoader: Validation dataloader
        torch.utils.data.DataLoader: Test dataloader
    """
    
    
    
    # Case 1: Work with the same preprocessing as for the original classifier test proposed by Claudius.
    # This means were working on the space of the datafiles.
    if postprocessing == True:
            
        # load the datafiles
        original_file = h5py.File(params["classification_set"], 'r')
        original = h5py_to_dict(original_file)
        original_file.close()
        
        for elem in original:
            num_samples = len(original[elem])
            break
        
        # Generate the samples. Pay attention, that the trainer dtype is temporarily the default dtype
        trainer_dtype = next(iter(test_trainer.model.parameters())).dtype
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(trainer_dtype)
        sample = test_trainer.generate(num_samples=num_samples, return_data=True, save_data=False, postprocessing=postprocessing)
        torch.set_default_dtype(old_dtype)
        
        # TODO: Solve differently, is not nice! This here mimics the effect of the save fuction. In the next change move this to the postprocessing or the generate step!
        sample["layer_0"] *= 1e3
        sample["layer_1"] *= 1e3
        sample["layer_2"] *= 1e3
        sample["energy"] *= 1e0
        
        # Drop the layers that are not needed, if requested
        if drop_layers is not None:
            assert 0==1, "should not be reachable for current implementation"

            for layer in drop_layers:
                original.pop(layer, None)
                
            for layer in drop_layers:
                sample.pop(layer, None)
        
        # Load the data files
        train_file, validation_file, test_file = split_dataset(merge_datasets(sample=sample, original=original), val_fraction, test_fraction)
        # Create the datasets
        dataset_train = CaloDataset(train_file, apply_logit=False, with_noise=False, return_label=True)
        dataset_val = CaloDataset(validation_file, apply_logit=False, with_noise=False, return_label=True)
        dataset_test = CaloDataset(test_file, apply_logit=False, with_noise=False, return_label=True)

    
    # Case 2: Work directly in the training space, directly on the output of the INN.
    # Should not happen for any ECAE case that is implemented right now
    elif postprocessing == False:
        assert 0==1, "This part should not be reachable"
        
        # Load the classification set using the data_util implementation
        # Device = CPU since we are needing them as numpy arrays...
        train_loader, _ = get_loaders(
                data_file_train=params.get("classification_set"),
                data_file_test=params.get('classification_set'),
                batch_size=params.get('batch_size'),
                device="cpu",
                width_noise=params.get("width_noise", 1e-7),
                use_extra_dim=params.get("use_extra_dim", False),
                use_extra_dims=params.get("use_extra_dims", False),
                layer=params.get("calo_layer", None)
            )
            
        # Only take the enery dimensions (cf. trainer)
        voxels = params["voxels"]
        voxels = voxels
        train_loader.data = train_loader.data[:, voxels]
        
        # Do computations on cpu using numpy
        real_data = train_loader.data.cpu().numpy()
        real_labels = np.ones(real_data.shape[0])
        
        # Generate the samples. Pay attention, that the trainer dtype is temporarily the default dtype
        trainer_dtype = next(iter(test_trainer.model.parameters())).dtype
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(trainer_dtype)
        generated_data = test_trainer.generate(real_data.shape[0], return_data=True, save_data=False, postprocessing=False)
        torch.set_default_dtype(old_dtype)
        
        generated_labels = np.zeros(generated_data.shape[0])
        
        # TODO: Make sure that np.append preserves the order?
        complete_data = np.append(real_data, generated_data, axis=0)
        complete_labels = np.append(real_labels, generated_labels)
        
        # Shuffle the data & labels
        shuffle_index = np.random.choice(complete_data.shape[0], complete_data.shape[0], replace=False)
        complete_data = complete_data[shuffle_index]
        complete_labels = complete_labels[shuffle_index]
        
        # Make tensors
        data = torch.tensor(complete_data).type(torch.get_default_dtype())
        labels = torch.tensor(complete_labels).type(torch.get_default_dtype())
        
        # Create the dataset
        dataset = TensorDataset(data, labels)
        
        # Calculate the number of samples for each set
        total_samples = len(dataset)
        val_samples = int(val_fraction * total_samples)
        test_samples = int(test_fraction * total_samples)
        train_samples = total_samples - val_samples - test_samples

        # Split the dataset into the train, test, and validation sets
        dataset_train, dataset_test, dataset_val = random_split(dataset, [train_samples, test_samples, val_samples])

        
    
    # Create the dataloaders
    batch_size = params.get("classifier_batch_size", 1000)
    
    # Keywords for the dataloader if using cuda
    kwargs = {'num_workers': 2, 'pin_memory': True} if 'cuda' in device else {}
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, **kwargs)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, **kwargs)
    
    return dataloader_train, dataloader_val, dataloader_test
