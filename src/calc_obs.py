import numpy as np

# def Etot_Einc_discrete(hlf):
#     target_energies = 2**np.linspace(8, 23, 16)
#     which_showers = ((hlf.Einc.squeeze() >= target_energies[i]) & (hlf.Einc.squeeze() < target_energies[i+1])).squeeze()
    
#     return hlf.GetEtot()[which_showers] / hlf.Einc.squeeze()[which_showers]

def Etot_Einc(hlf):
    return hlf.GetEtot() / hlf.Einc.squeeze()

def E_layers(hlf, layer):
    return hlf.GetElayers()[layer]

def ECEtas(hlf, layer):
    return hlf.GetECEtas()[layer]
    
def ECPhis(hlf, layer):
    return hlf.GetECPhis()[layer]

def ECWidthEtas(hlf, layer):
    return hlf.GetWidthEtas()[layer]

def ECWidthPhis(hlf, layer):
    return hlf.GetWidthPhis()[layer]

def ECRads(hlf, layer):
    return hlf.GetECRads()[layer]

def ECWidthRads(hlf, layer):
    return hlf.GetWidthRads()[layer]

def cell_dist(hlf):
    return hlf.showers.flatten()

def cell_dist_by_layer(hlf, layer=None):
    
    if layer is not None:
        return hlf.showers[:, hlf.bin_edges[layer]:hlf.bin_edges[layer+1]].flatten()
    
    # TODO: Maybe recursive is nicer...
    layer_dist = {}
    for layer in hlf.relevantLayers:
        layer_dist[layer] = hlf.showers[:, hlf.bin_edges[layer]:hlf.bin_edges[layer+1]].flatten()
    return layer_dist

def calc_brightest_voxel(hlf, layer=None, N=1):
    if layer is not None:
        layer_data = hlf.showers[:, hlf.bin_edges[layer]:hlf.bin_edges[layer+1]]
        layer_data.sort(axis=1)
        N = min(N, layer_data.shape[1])
        return layer_data[:,-N]/layer_data.sum(axis=1)
    
    brightest_voxels = {}
    for layer in hlf.relevantLayers:
        brightest_voxels[layer] = calc_brightest_voxel(hlf, layer=layer, N=N)
        
    return brightest_voxels

def sparsity(hlf, layer):
    return hlf.GetSparsity()[layer]