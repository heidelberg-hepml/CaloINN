import numpy as np
import h5py
import torch
from copy import deepcopy
from myDataLoader import MyDataLoader
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from calc_obs import calc_sparsity

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
    
    # Also add the sparsity as a condtion
    funcs = [calc_sparsity, calc_sparsity, calc_sparsity]
    params = [{'layer': 0}, {'layer': 1}, {'layer': 2}]
    
    more_conditions = get_further_conditions(funcs, data, params)[binary_mask]    
    
    # Also add the centroids as condition (here x is already masked!)
    more_conditions = (more_conditions.T).tolist()
    for dir in ["phi", "eta"]:
        means = calc_centroids(data=x, dir=dir, numpy=True)
        for mean in means:
            more_conditions.append(mean)
    more_conditions = np.array(more_conditions).T
    
    # Append the new conditions between the energy conditions and the true, unnormalized layer energies
    c = np.concatenate((c[:, :4], more_conditions, c[:, 4:]), axis=1)

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
    
    # Save the actual layer energies for numerical stability!
    return np.concatenate((e_part, u1, u2, u3, e0, e1, e2), axis=1)

def calc_centroids(data, dir, numpy=False):
    
    def get_bin_centers(dir, data, layer, numpy):
        
        shape_layer_0, shape_layer_1, shape_layer_2 = get_layer_shapes(data_flattened=data)
        
        if dir == 'phi':
            cells = (shape_layer_0[0], shape_layer_1[0], shape_layer_2[0])[layer]
        elif dir == 'eta':
            cells = (shape_layer_0[1], shape_layer_1[1], shape_layer_2[1])[layer]
        else:
            raise ValueError(f"dir={dir} not in ['eta', 'phi']")
        bins = np.linspace(0, 1, cells + 1)
        
        if not numpy:
            bins = torch.from_numpy(bins).to(data.device)
            
        return (bins[1:] + bins[:-1]) / 2.
    
    # Reshape the data array to the original shape
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data)
    shape_layer_0, shape_layer_1, shape_layer_2 = get_layer_shapes(data_flattened=data)
    
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1
    
    layer_0 = data[:, :l_0].reshape(-1, shape_layer_0[0], shape_layer_0[1])
    layer_1 = data[:, l_0:l_01].reshape(-1, shape_layer_1[0], shape_layer_1[1])
    layer_2 = data[:, l_01:].reshape(-1, shape_layer_2[0], shape_layer_2[1])

    means = []
    
    for layer_index, layer in enumerate([layer_0, layer_1, layer_2]):
    
        bin_centers = get_bin_centers(dir, data, layer_index, numpy)
        
        # In order to handle numpy arrays and pytorch tensors
        if numpy:
            energies = np.sum(layer, axis=(1, 2))
        else:
            energies = torch.sum(layer, axis=(1, 2)).to(data.device)

        if dir == 'phi':
            value = bin_centers.reshape(-1, 1)
        elif dir == 'eta':
            value = bin_centers.reshape(1, -1)

        if numpy:
            mean = np.sum(layer * value, axis=(1, 2))/(energies+1e-10)
        else:
            mean = torch.sum(layer * value, axis=(1, 2))/(energies+1e-10)
        # std = np.sqrt(np.sum(layer * (value - mean[:,None, None])**2, axis=(1, 2))/(energies+1e-10))
        
        means.append(mean)

    return means

def get_further_conditions(funcs, data, params):
    
    conds = []
    
    for i, func in enumerate(funcs):
        conds.append(func(deepcopy(data), **(params[i])))
        
    return np.array(conds).T

def normalize_layers(data_flattened, conditions):
    
    # Prevent inplace operations
    data_flattened = torch.clone(data_flattened)
    conditions = torch.clone(conditions)
    
    # Get the layer sizes
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data_flattened)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1    
    
    # Use the exact layer energies for numerical stability
    e0 = conditions[:, [-3]]
    e1 = conditions[:, [-2]]
    e2 = conditions[:, [-1]]

    # Normalize each layer by the layer energy
    data_flattened[..., :l_0] = data_flattened[..., :l_0] / (e0 + 1.e-7)
    data_flattened[..., l_0:l_01] = data_flattened[..., l_0:l_01] / (e1 + 1e-7)
    data_flattened[..., l_01:] = data_flattened[..., l_01:] / (e2 + 1e-7)
    
    return data_flattened

def unnormalize_layers(data_flattened, conditions, clone=True):
    # Prevent inplace operations
    
    if clone:
        output = torch.clone(data_flattened)
        conditions = torch.clone(conditions)
        
    output = torch.zeros_like(data_flattened).to(data_flattened.device)
    # Get the layer sizes
    size_layer_0, size_layer_1, size_layer_2 = get_layer_sizes(data_flattened=data_flattened)
    l_0 = size_layer_0
    l_01 = size_layer_0 + size_layer_1 
    
    # Use the exact layer energies for numerical stability
    e0 = conditions[:, [-3]]
    e1 = conditions[:, [-2]]
    e2 = conditions[:, [-1]]
    
    # Normalize the layers to the correct energies
    output[..., :l_0]    = data_flattened[..., :l_0] / (torch.sum(data_flattened[..., :l_0], axis=1, keepdims=True) + 1e-7) * e0
    output[..., l_0:l_01] = data_flattened[..., l_0:l_01] / (torch.sum(data_flattened[..., l_0:l_01], axis=1, keepdims=True) + 1e-7) * e1
    output[..., l_01:]    = data_flattened[..., l_01:] / (torch.sum(data_flattened[..., l_01:], axis=1, keepdims=True) + 1e-7) * e2
    return output

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
