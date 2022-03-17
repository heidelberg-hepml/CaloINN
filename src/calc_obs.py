import numpy as np
import h5py

def load_data(data_file):
    full_file = h5py.File(data_file, 'r')
    layer_0 = np.float32(full_file['layer_0'][:] / 1e3)
    layer_1 = np.float32(full_file['layer_1'][:] / 1e3)
    layer_2 = np.float32(full_file['layer_2'][:] / 1e3)
    energy = np.float32(full_file['energy'][:] /1e0)
    full_file.close()

    data = {
        'layer_0': layer_0,
        'layer_1': layer_1,
        'layer_2': layer_2,
        'energy': energy[:,0]
    }

    return data

def apply_mask(data, mask):
    layer_0 = data['layer_0']
    layer_1 = data['layer_1']
    layer_2 = data['layer_2']
    energy = data['energy']

    return {
        'layer_0': layer_0[mask],
        'layer_1': layer_1[mask],
        'layer_2': layer_2[mask],
        'energy': energy[mask]
    }

def calc_e_layer(data, layer=0):
    return np.sum(data[f'layer_{layer}'],axis=(1,2))

def calc_e_detector(data):
    return np.sum(data['layer_0'],(1,2)) + np.sum(data['layer_1'],(1,2)) + np.sum(data['layer_2'],(1,2))

def calc_e_parton(data):
    return data['energy']

def calc_e_ratio(data):
    e_detector = calc_e_detector(data)
    e_parton = data['energy']
    return e_detector/e_parton

def calc_e_layer_normd(data, layer=0):
    e_layer = calc_e_layer(data, layer)
    e_total = calc_e_detector(data)
    return e_layer/e_total

def calc_brightest_voxel(data, layer=0, N=1):
    layer = data[f'layer_{layer}']
    layer = layer.reshape((layer.shape[0], -1))
    layer = layer[layer.sum(axis=1)>0]
    layer.sort(axis=1)
    return layer[:,-N]/layer.sum(axis=1)

def get_bin_centers(layer, dir):
    if dir == 'phi':
        cells = (3, 12, 12)[layer]
    elif dir == 'eta':
        cells = (96, 12, 6)[layer]
    else:
        raise ValueError(f"dir={dir} not in ['eta', 'phi']")
    bins = np.linspace(-240, 240, cells + 1)
    return (bins[1:] + bins[:-1]) / 2.

def calc_centroid(data, layer=0, dir='phi'):
    bin_centers = get_bin_centers(layer, dir)

    layer = data[f'layer_{layer}']
    energies = layer.sum(axis=(1, 2))

    if dir == 'phi':
        value = bin_centers.reshape(-1, 1)
    elif dir == 'eta':
        value = bin_centers.reshape(1, -1)

    mean = np.sum(layer * value, axis=(1, 2))/(energies+1e-10)
    std = np.sqrt(np.sum((layer * value - mean[:,None, None])**2, axis=(1, 2))/(energies+1e-10))

    return mean, std

def calc_centroid_mean(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[0]

def calc_centroid_std(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[1]

def calc_layer_diff(data, layer1=0, layer2=1, dir='phi'):
    mean_1, _ = calc_centroid(data, layer1, dir)
    mean_2, _ = calc_centroid(data, layer2, dir)
    return mean_2 - mean_1

def calc_sparsity(data, layer=0, threshold=1e-5):
    layer = data[f'layer_{layer}']
    return (layer > threshold).mean((1, 2))

def calc_spectrum(data, layer=0, threshold=1e-5):
    data = data[f'layer_{layer}']
    return data[data > threshold]

def calc_coro(data, layer=0, threshold=1e-5):
    img = data[f'layer_{layer}']
    mask = img > threshold
    img[~mask] = 0.
    img /= np.sum(img, axis=(1,2), keepdims=True)
    xpixels = get_bin_centers(layer, 'eta')
    ypixels = get_bin_centers(layer, 'phi')

    coro = np.zeros_like(img)
    for i in range(len(ypixels)):
        for j in range(len(xpixels)):
            x, y = np.meshgrid(xpixels-xpixels[j], ypixels-ypixels[i])
            R = np.sqrt(x*x+y*y)
            mask_l = mask[:,i,j]
            coro[mask_l,:,:] += img[mask_l,:,:]*img[mask_l,i,j][:,None,None]*R[None,:,:]**0.2
    return np.sum(coro, axis=(1,2))

