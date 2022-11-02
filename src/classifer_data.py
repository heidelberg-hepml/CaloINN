""" Dataloader for calorimeter data.
    Inspired by https://github.com/kamenbliznashki/normalizing_flows

    Used for
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import os
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

ALPHA = 1e-6
def logit(x):
    return np.log(x / (1.0 - x))

def logit_trafo(x):
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)


class CaloDataset(Dataset):
    """CaloGAN dataset of [2]."""

    def __init__(self, data_path,
                 transform_0=None, transform_1=None, transform_2=None,
                 apply_logit=True, with_noise=False,
                 return_label=False):
        """
        Args:
            data_path (string): path to .hdf5 file
            transform_i (callable, optional): Optional transform to be applied
            on data of layer i
        """


        self.full_file = h5py.File(data_path, 'r')

        self.apply_logit = apply_logit
        self.with_noise = with_noise

        self.transform_0 = transform_0
        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.return_label = return_label

        self.input_dims = {'0': (3, 96), '1': (12, 12), '2': (12, 6)}
        self.input_size = {'0': 288, '1': 144, '2': 72}

        # normalizations to 100 GeV
        self.file_layer_0 = self.full_file['layer_0'][:] / 1e5
        self.file_layer_1 = self.full_file['layer_1'][:] / 1e5
        self.file_layer_2 = self.full_file['layer_2'][:] / 1e5
        self.file_energy = self.full_file['energy'][:] /1e2
        self.file_overflow = self.full_file['overflow'][:] /1e5
        if self.return_label:
            self.file_label = self.full_file['label'][:]
        self.full_file.close()

    def __len__(self):
        # assuming file was written correctly
        #return len(self.full_file['energy'])
        return len(self.file_energy)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## normalizations to 100 GeV
        #layer_0 = self.full_file['layer_0'][idx] / 1e5
        #layer_1 = self.full_file['layer_1'][idx] /1e5
        #layer_2 = self.full_file['layer_2'][idx] /1e5
        #energy = self.full_file['energy'][idx] /1e2
        #overflow = self.full_file['overflow'][idx] /1e5
        layer_0 = self.file_layer_0[idx]
        layer_1 = self.file_layer_1[idx]
        layer_2 = self.file_layer_2[idx]
        energy = self.file_energy[idx]
        overflow = self.file_overflow[idx]

        if self.with_noise:
            layer_0 = add_noise(layer_0)
            layer_1 = add_noise(layer_1)
            layer_2 = add_noise(layer_2)

        layer_0_E = layer_0.sum(axis=(-1, -2), keepdims=True)
        layer_1_E = layer_1.sum(axis=(-1, -2), keepdims=True)
        layer_2_E = layer_2.sum(axis=(-1, -2), keepdims=True)

        if self.transform_0:
            if self.transform_0 == 'E_norm':
                layer_0 = layer_0 / energy
            elif self.transform_0 == 'L_norm':
                layer_0 = layer_0 / (layer_0_E + 1e-16)
            else:
                layer_0 = self.transform_0(layer_0)

        if self.transform_1:
            if self.transform_1 == 'E_norm':
                layer_1 = layer_1 / energy
            elif self.transform_1 == 'L_norm':
                layer_1 = layer_1 / (layer_1_E + 1e-16)
            else:
                layer_1 = self.transform_1(layer_1)

        if self.transform_2:
            if self.transform_2 == 'E_norm':
                layer_2 = layer_2 / energy
            elif self.transform_2 == 'L_norm':
                layer_2 = layer_2 / (layer_2_E + 1e-16)
            else:
                layer_2 = self.transform_2(layer_2)

        if self.apply_logit:
            layer_0 = logit_trafo(layer_0)
            layer_1 = logit_trafo(layer_1)
            layer_2 = logit_trafo(layer_2)

        sample = {'layer_0': layer_0, 'layer_1': layer_1,
                  'layer_2': layer_2, 'energy': energy,
                  'overflow': overflow, 'layer_0_E': layer_0_E.squeeze(),
                  'layer_1_E': layer_1_E.squeeze(), 'layer_2_E': layer_2_E.squeeze()}
        if self.return_label:
            #sample['label'] = self.full_file['label'][idx]
            sample['label'] = self.file_label[idx]

        return sample


# TODO: Added the data_path_val parameter
def get_dataloader(data_path_train, data_path_test, device, data_path_val=None,
                   batch_size=32, apply_logit=True, with_noise=False, normed=False,
                   normed_layer=False, return_label=False):

    if normed and normed_layer:
        raise ValueError("Cannot normalize data to layer and event simultaenously")

    kwargs = {'num_workers': 2, 'pin_memory': True} if 'cuda' in device else {}

    if normed:
        dataset_kwargs = {'transform_0': 'E_norm',
                          'transform_1': 'E_norm',
                          'transform_2': 'E_norm',
                          'with_noise': with_noise}
    elif normed_layer:
        dataset_kwargs = {'transform_0': 'L_norm',
                          'transform_1': 'L_norm',
                          'transform_2': 'L_norm',
                          'with_noise': with_noise}
    else:
        dataset_kwargs = {'with_noise': with_noise}

    train_dataset = CaloDataset(data_path_train, apply_logit=apply_logit,
                                return_label=return_label, **dataset_kwargs)
    if data_path_val is not None:
        val_dataset = CaloDataset(data_path_val, apply_logit=apply_logit,
                                    return_label=return_label, **dataset_kwargs)
    test_dataset = CaloDataset(data_path_test, apply_logit=apply_logit,
                                return_label=return_label, **dataset_kwargs)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, **kwargs)
    if data_path_val is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, **kwargs)
    
    if data_path_val is not None:
        return train_dataloader, val_dataloader, test_dataloader    
    return train_dataloader, test_dataloader

def add_noise(input_tensor):
    noise = np.random.rand(*input_tensor.shape)*1e-8
    return input_tensor+noise

def save_samples_to_file(samples, energies, filename, threshold):
    """ saves the given sample to hdf5 file, like training data
        add 0s to overflow to match structure of training data
    """

    assert len(energies) == len(samples)

    data = samples.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < threshold, np.zeros_like(data), data)

    energies = energies.to('cpu').unsqueeze(-1).numpy()*1e2
    overflow = np.zeros((len(energies), 3))
    layer_0 = data[..., :288].reshape(-1, 3, 96)
    layer_1 = data[..., 288:432].reshape(-1, 12, 12)
    layer_2 = data[..., 432:].reshape(-1, 12, 6)

    save_file = h5py.File(filename, 'w')

    save_file.create_dataset('layer_0', data=layer_0)
    save_file.create_dataset('layer_1', data=layer_1)
    save_file.create_dataset('layer_2', data=layer_2)
    save_file.create_dataset('energy', data=energies)
    save_file.create_dataset('overflow', data=overflow)

    save_file.close()
