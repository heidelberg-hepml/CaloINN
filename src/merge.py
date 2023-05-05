import os
import numpy as np
import h5py
import argparse

def merge(file1, file2, new_file_name, ratio, N=100000):
    print(file1, file2, new_file_name, ratio, N)
    input_file_1 = h5py.File(file1, 'r')
    input_file_2 = h5py.File(file2, 'r')
    new_file = h5py.File(new_file_name, 'w')

    N1 = int(ratio*N)
    N2 = N-N1
    #perm = np.random.permutation(N)
    print(N1, N2, N)

    for key in input_file_1.keys():
        data1 = input_file_1[key] #[:N1]
        data2 = input_file_2[key] #[:N2]
        data = np.concatenate([data1, data2])
        new_file.create_dataset(key, data=data) #[perm])

    new_file.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file1', help='File 1 to be merged')
    parser.add_argument('--file2', help='File 2 to be merged')
    parser.add_argument('--new_file', help='New file')
    parser.add_argument('--ratio', type=float, default=0.5)

    args = parser.parse_args()

    merge(args.file1, args.file2, args.new_file, args.ratio)

if __name__=='__main__':
    main()
