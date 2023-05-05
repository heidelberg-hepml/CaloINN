""" 
	takes .hdf5 file as input and splits it in 2 smaller files for training and testing

    Used for:
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', help='File to be split')
parser.add_argument('--test_fraction', '-f', default=0.7, type=float,
                    help='Fraction of events to go to the test set')

args = parser.parse_args()

input_file = h5py.File(args.file, 'r')

key_len = []
for key in input_file.keys():
    key_len.append(len(input_file[key]))
key_len = np.array(key_len)

assert np.all(key_len==key_len[0])

cut_index = int(args.test_fraction * key_len[0])

train_file = h5py.File('train_'+args.file, 'w')
test_file = h5py.File('test_'+args.file, 'w')

for key in input_file.keys():
    train_file.create_dataset(key, data=input_file[key][:cut_index])
    test_file.create_dataset(key, data=input_file[key][cut_index:])

train_file.close()
test_file.close()


