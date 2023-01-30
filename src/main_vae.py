from functools import partial
import shutil
import argparse
import os
import sys
import yaml
import torch
import os

from documenter import Documenter
from trainer import VAETrainer, DNNTrainer

from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='where to find the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'

    # set default parameters for the file locations using the particle type parameter
    if "data_path_train" not in params:
        particle = params.get("particle_type", "piplus")
        data_path_train = os.path.join("..", "..", "..", "Datasets", particle , "train_" + particle + ".hdf5")
        params["data_path_train"] = data_path_train
        
    if "data_path_test" not in params:
        particle = params.get("particle_type", "piplus")
        data_path_test = os.path.join("..", "..", "..", "Datasets", particle , "test_" + particle + ".hdf5")
        params["data_path_test"] = data_path_test

    if "classification_set" not in params:
        particle = params.get("particle_type", "piplus")
        classification_set = os.path.join("..", "..", "..", "Datasets", particle , "cls_" + particle + ".hdf5")
        params["classification_set"] = classification_set

    # Initialize the documenter class
    # Sends all outputs to a log file and manages the file system of the output folder
    doc = Documenter(params['run_name'], block_name=params.get("block_name", None))
    
    # Backup of the parameter file
    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    
    
    print('device: ', device)
    print('commit: ', os.popen(r'git rev-parse --short HEAD').read(), end='')

    # Set the default dtype
    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)
        

    # train the model
    sys.stdout.flush()
    vae_trainer = VAETrainer(params, device, doc)
    print("Start Training:")
    vae_trainer.train()


if __name__=='__main__':
    main()
