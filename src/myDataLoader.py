import math
from typing import Tuple, Iterator

import torch

class MyDataLoader:
    """ A data loader that is way faster than the PyTorch data loader for data with a dimension of less than tausend. """

    data: torch.Tensor
    batch_size: int
    drop_last: bool
    shuffle: bool

    def __init__(self, data: torch.Tensor, cond: torch.Tensor, batch_size: int, drop_last:bool=False, shuffle:bool=True) -> None:
        """
            Initializes MyDataLoader class.

            Parameters:
            data (tensor): Data
            cond (tensor): Condition
            batch_size (int): Batch size
            drop_last (bool): If true the last batch is dropped if it is smaller as the given batch size
            shuffle (bool): Weather or not to shuffle the data
            width_noise (float): Width of the noise to add to the data
        """
        self.data = data
        self.cond = cond
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        self.vae_resampling = False
        
        if self.drop_last:
            self.max_batch = len(self.data) // self.batch_size
        else:
            self.max_batch = math.ceil(len(self.data) / self.batch_size)

    def drop_last_batch(self) -> None:
        self.drop_last = True
        self.max_batch = len(self.data) // self.batch_size

    def activate_vae_resampling(self) -> None:
        
        if self.vae_resampling == False:
            
            assert (self.data.shape[1] - 3) % 2 == 0, "The latent space cannot be partitioned equally into mu and sigma!"
            size_latent_space = (self.data.shape[1] - 3) // 2
            print(f"Assuming latent space of dimension {size_latent_space}.")
            self.mu = torch.clone(self.data[:, :size_latent_space])
            self.logvar = torch.clone(self.data[:, size_latent_space:-3])
            self.energy_dims = torch.clone(self.data[:, -3:])
            
            std = torch.exp(0.5*self.logvar)
            eps = torch.randn_like(std)
            
            self.data = torch.cat((eps * std + self.mu, self.energy_dims), axis=1)
        
        self.vae_resampling = True

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        if self.shuffle:
            self.index = torch.randperm(len(self.data), device=self.data.device)
        else:
            self.index = torch.arange(len(self.data), device=self.data.device)
        self.batch = 0

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch >= self.max_batch:
            raise StopIteration
        first = self.batch*self.batch_size
        last = min(first+self.batch_size, len(self.data))
        idx = self.index[first:last]
        self.batch += 1
        
        if not self.vae_resampling:
        
            data = torch.clone(self.data[idx])
            cond = torch.clone(self.cond[idx])
            
        else:
            
            std = torch.exp(0.5*self.logvar[idx])
            eps = torch.randn_like(std)
            
            data = torch.cat((eps * std + self.mu[idx], self.energy_dims[idx]), axis=1)
            cond = torch.clone(self.cond[idx])
        
        return data, cond
