import numpy as np

def return_voxel(data_dict, voxel_index=0):
    data = data_dict["data"]
    return data[:, voxel_index]

def calc_e_layer(data, layer=0):
    return np.sum(data[f'layer_{layer}'],axis=(1,2))

def calc_e_detector(data):
    return np.sum(data['layer_0'],(1,2)) + np.sum(data['layer_1'],(1,2)) + np.sum(data['layer_2'],(1,2))

def calc_e_parton(data):
    return np.copy(data['energy'][:,0])

def calc_e_ratio(data):
    e_detector = calc_e_detector(data)
    e_parton = calc_e_parton(data)
    return e_detector/e_parton

def calc_e_layer_normd(data, layer=0):
    e_layer = calc_e_layer(data, layer)
    e_total = calc_e_detector(data)
    return e_layer/e_total

def calc_brightest_voxel(data, layer=0, N=1):
    layer = np.copy(data[f'layer_{layer}'])
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
    energies = np.sum(layer, axis=(1, 2))

    if dir == 'phi':
        value = bin_centers.reshape(-1, 1)
    elif dir == 'eta':
        value = bin_centers.reshape(1, -1)

    mean = np.sum(layer * value, axis=(1, 2))/(energies+1e-10)
    std = np.sqrt(np.sum(layer * (value - mean[:,None, None])**2, axis=(1, 2))/(energies+1e-10))

    return mean, std

def calc_centroid_mean(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[0]

def calc_centroid_std(data, layer=0, dir='phi'):
    return calc_centroid(data, layer, dir)[1]

def calc_layer_diff(data, layer1=0, layer2=1, dir='phi'):
    mean_1 = calc_centroid_mean(data, layer1, dir)
    mean_2 = calc_centroid_mean(data, layer2, dir)
    return mean_2 - mean_1

def calc_sparsity(data, layer=0, threshold=1e-5):
    layer = data[f'layer_{layer}']
    return (layer > threshold).mean((1, 2))

def calc_spectrum(data, layer=0, threshold=1e-5):
    data = data[f'layer_{layer}']
    return data[data > threshold]

def calc_depth_weighted_total_energy(data):
    l_1 = data[f'layer_1']*1e3
    l_2 = data[f'layer_2']*1e3
    w_en = l_1.sum((1,2)) + 2*l_2.sum((1,2))
    return w_en

def calc_depth_weighted_total_energy_normed(data):
    w_en_normd = calc_depth_weighted_total_energy(data)/(calc_e_detector(data)*1e3)
    return w_en_normd

def calc_depth_weighted_total_energy_std(data):
    en_1 = data[f'layer_1'].sum((1,2))*1e3
    en_2 = data[f'layer_2'].sum((1,2))*1e3
    tot = calc_e_detector(data)*1e3
    w_en_normd_std = layer_std(en_1, en_2, tot) 
    return w_en_normd_std

def calc_layer_brightest_ratio(data, layer=0):
    brightest_1 = calc_brightest_voxel(data, layer=layer, N=1)
    brightest_2 = calc_brightest_voxel(data, layer=layer, N=2)
    ratio = (brightest_1 - brightest_2)/(brightest_1 + brightest_2)
    return ratio

def layer_std(layer1, layer2, total):
    """ helper function for standard deviation of layer depth"""
    term1 = (layer1 + 4.*layer2) / total
    term2 = ((layer1 + 2.*layer2)/total)**2
    return np.sqrt(term1-term2)

def calc_coro(data, layer=0, threshold=1e-5):
    img = np.copy(data[f'layer_{layer}'])
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
