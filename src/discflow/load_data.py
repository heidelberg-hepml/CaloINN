import numpy as np
import pandas as pd
import h5py

from observables import ObsE, ObsPx, ObsPy, ObsPz

def load_h5(path):
    try:
        data = pd.read_hdf(path).to_numpy()
    except:
        f = h5py.File(path, "r")
        try:
            data = np.array(f[list(f.keys())[0]])
            if data.ndim <= 1:
                raise(RuntimeError("Wrong data format"))
        except RuntimeError as error:
            data = np.array(f[list(f.keys())[0]]["table"])
            data = [event[1] for event in data]
    data = np.delete(data, [302775,682515,3299941,4749733,2669734], axis = 0)
    ret_data = data.copy()

    return ret_data

def split_data(data, test_split):
    split_index = int(len(data)*test_split)
    return data[split_index:,:], data[:split_index,:]

def get_eppp_observables(data):
    observables = {}
    for i in range(data.shape[1] // 4):
        observables.update({
            ObsE(i):  data[:,4*i+0],
            ObsPx(i): data[:,4*i+1],
            ObsPy(i): data[:,4*i+2],
            ObsPz(i): data[:,4*i+3]
        })
    return observables

def select_by_jet_count(data, min_jets, max_jets):
    n_jets = np.sum(data[:,0::4] != 0., axis=1)
    return data[(n_jets >= min_jets) & (n_jets <= max_jets), :4*max_jets]
