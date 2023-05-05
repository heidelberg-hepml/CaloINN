""" takes two .hdf5 file as input and merges/shuffles them for classifier input

    Used for:
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import os
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file1', help='File 1 to be merged (truth label 0)')
parser.add_argument('--file2', help='File 2 to be merged (truth label 1)')

args = parser.parse_args()

input_file_1 = h5py.File(args.file1, 'r')
input_file_2 = h5py.File(args.file2, 'r')

key_len_1 = []
for key in input_file_1.keys():
    key_len_1.append(len(input_file_1[key]))
key_len_1 = np.array(key_len_1)
assert np.all(key_len_1 == key_len_1[0])
key_len_2 = []
for key in input_file_2.keys():
    key_len_2.append(len(input_file_2[key]))
key_len_2 = np.array(key_len_2)
assert np.all(key_len_2 == key_len_2[0])

assert np.all(key_len_1 == key_len_2)

file1_name = os.path.splitext(os.path.basename(args.file1))[0]
file2_name = os.path.splitext(os.path.basename(args.file2))[0]

new_file_name = 'merged_'+ file1_name + '_' + file2_name + '.hdf5'
new_file = h5py.File(new_file_name, 'w')

shuffle_order = np.arange(key_len_1[0]+key_len_2[0])
np.random.shuffle(shuffle_order)


for key in input_file_1.keys():
    data1 = input_file_1[key][:]
    data2 = input_file_2[key][:]
    data = np.concatenate([data1, data2])
    new_file.create_dataset(key, data=data[shuffle_order])

truth1 = np.zeros(key_len_1[0])
truth2 = np.ones(key_len_2[0])
truth = np.concatenate([truth1, truth2])
new_file.create_dataset('label', data=truth[shuffle_order])

new_file.close()


