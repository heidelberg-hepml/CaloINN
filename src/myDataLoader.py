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
        
        if self.drop_last:
            self.max_batch = len(self.data) // self.batch_size
        else:
            self.max_batch = math.ceil(len(self.data) / self.batch_size)

    def drop_last_batch(self) -> None:
        self.drop_last = True
        self.max_batch = len(self.data) // self.batch_size

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
        
        data = torch.clone(self.data[idx])
        cond = torch.clone(self.cond[idx])
        
        return data, cond
