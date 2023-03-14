import h5py
import numpy as np
from detector_specs import _photon_detector_spec, _pion_detector_spec

with h5py.File("/remote/gpu05/lindenberg/ChallengeDatasets/Dataset1/dataset_1_photons_1.hdf5", "r") as f: 
    photon_energies = f['incident_energies'][()]; 
    photon_showers  = f['showers'][()]; 

with h5py.File("/remote/gpu05/lindenberg/ChallengeDatasets/Dataset1/dataset_1_pions_1.hdf5", "r") as f: 
    pion_energies   = f['incident_energies'][()]; 
    pion_showers    = f['showers'][()]; 

print("Photon Energies:", photon_energies.shape); 
print("Photon Showers:", photon_showers.shape); 
print("Pion Energies:", pion_energies.shape); 
print("Pion Showers:", pion_showers.shape); 


def get_sample(*, data, idx, detector): 
    sample = {}; 
    slice_start = 0;   
    slice_end   = 0;    
    for layer_key in detector["layers"]: 
        layer = detector["layers"][layer_key]; 
        
        slice_end += layer["num_angular"] * layer["num_radial"]; 

        slice = data[idx, slice_start:slice_end]; 
#        sample[layer_key] = slice; 
        sample[layer_key] = slice.reshape(layer["num_angular"], layer["num_radial"]); 

        slice_start = slice_end; 
    return sample; 


def get_photon_sample(idx): 
    sample = {}; 
    slice_start = 0; 
    slice_end   = 0; 

    for layer_key in _photon_detector_spec["layers"]: 
        layer = _photon_detector_spec["layers"][layer_key]; 

        slice_end += layer["num_angular"] * layer["num_radial"]; 
        slice = photon_showers[idx, slice_start:slice_end]; 

#        sample[layer_key] = slice; 
        sample[layer_key] = slice.reshape(layer["num_angular"], layer["num_radial"]); 

        slice_start = slice_end; 
    return sample; 

def get_pion_sample(idx): 
    sample = {}; 
    slice_start = 0; 
    slice_end   = 0; 

    for layer_key in _pion_detector_spec["layers"]: 
        layer = _pion_detector_spec["layers"][layer_key]; 

        slice_end += layer["num_angular"] * layer["num_radial"]; 
        slice = pion_showers[idx, slice_start:slice_end]; 

#        sample[layer_key] = slice; 
        sample[layer_key] = slice.reshape(layer["num_angular"], layer["num_radial"]); 

        slice_start = slice_end; 
    return sample; 


#ex = get_sample(
#    data        = photon_showers, 
#    idx         = 0, 
#    detector    = _photon_detector_spec
#); 

#ex = get_photon_sample(0); 
ex = get_pion_sample(0); 

total = 0; 
for layer_key in ex: 
    total += len(ex[layer_key].reshape(-1)); 
    print(layer_key, ex[layer_key].shape);
print("Total:", total); 




