from tqdm import tqdm
import observables
import torch
import numpy as np

def tqdm_verbose(iterable, verbose, **kwargs):
    if verbose:
        return tqdm(iterable, **kwargs)
    else:
        return iterable

def tqdm_write_verbose(text, verbose):
    if verbose:
        tqdm.write(text)
    else:
        print(text, flush=True)

class FunctionRegistry:
    def __init__(self):
        self.functions = {}

    def __call__(self, func):
        self.functions[func.__name__] = func
        return func

model_class = FunctionRegistry()

def eval_observables_list(param):
    obs_all = { name: value for name, value in vars(observables).items()
                if name[:3] == "Obs" or name in ["JetSum", "InclJetSum"]}
    if isinstance(param, str):
        return eval(param, obs_all, {})
    elif isinstance(param, list):
        return [eval_observables_list(expr) for expr in param]
    else:
        raise ValueError("Invalid type of the list of observables")

def eval_observables_expr(expr, data):
    obs_all = { name: value for name, value in vars(observables).items()
                if name[:3] == "Obs" or name in ["JetSum", "InclJetSum"]}
    obs_all["d"] = lambda obs: obs.from_data(data)
    return eval(expr, obs_all, {})


class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, label, cat_count, upsampling):
        self.categories = [torch.where(label == i)[0] for i in range(cat_count)]
        self.counts = [len(cat) for cat in self.categories]
        self.upsampling = upsampling
        if upsampling:
            self.max_count = max(self.counts)
            self.length = len(self.categories) * self.max_count
        else:
            self.min_count = min(self.counts)
            self.length = len(self.categories) * self.min_count

    def __iter__(self):
        if self.upsampling:
            indices = [torch.cat([
                cat.repeat_interleave(self.max_count // count),
                cat[torch.randperm(len(cat))[:self.max_count % count]]
            ]) for count, cat in zip(self.counts, self.categories)]
        else:
            indices = [cat[torch.randperm(len(cat))[:self.min_count]]
                           for cat in self.categories]
        indices = torch.cat(indices)
        yield from indices[torch.randperm(len(indices))].tolist()

    def __len__(self):
        return self.length

def zero_nan_events(data):
    '''Set all NaNs in input tensor to zero.
        args:
            data    : [tensor] tensor holding the observable for each event (row: events, column: observables).
    '''

    mask        = (data != data).nonzero(as_tuple=True)
    n_zero      = mask[0].shape[0]
    data[mask]  = 0

    return n_zero, data
