import math
import torch
from torch.distributions import Beta

class MyDataLoader:

    data: torch.Tensor
    batch_size: int
    drop_last: bool
    shuffle: bool

    def __init__(self, data: torch.Tensor, cond: torch.Tensor, batch_size: int,
                drop_last:bool=False, shuffle:bool=True, width_noise:float=1e-7):
        self.data = data
        self.cond = cond
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.width_noise  = width_noise
        self.deta_distribution = Beta(torch.tensor([3.], device=data.device), torch.tensor([3.], device=data.device))

        if self.drop_last:
            self.max_batch = len(self.data) // self.batch_size
        else:
            self.max_batch = math.ceil(len(self.data) / self.batch_size)

        self.initialize()

    def add_noise(self, input):
        noise = self.deta_distribution.sample(input.shape)*self.width_noise
        return input + noise[...,0]

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self):
        return self

    def initialize(self):
        if self.shuffle:
            self.index = torch.randperm(len(self.data), device=self.data.device)
        else:
            self.index = torch.arange(len(self.data), device=self.data.device)
        self.batch = 0

    def __next__(self):
        if self.batch >= self.max_batch:
            self.initialize()
            raise StopIteration
        first = self.batch*self.batch_size
        last = min(first+self.batch_size, len(self.data))
        idx = self.index[first:last]
        self.batch += 1
        data = torch.clone(self.add_noise(self.data[idx]))
        cond = torch.clone(self.cond[idx])
        return data, cond
