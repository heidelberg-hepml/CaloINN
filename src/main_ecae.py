from functools import partial
import shutil
import argparse
import os
import sys
import yaml
import torch
import os

from documenter import Documenter
from trainer import ECAETrainer, DNNTrainer

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
    if "data_path" not in params:
        particle_type = params.get("particle_type", "pion")
        data_path = f"/remote/gpu06/ernst/Master_Thesis/Datasets/Dataset1/dataset_1_{particle_type}s_1.hdf5"
        params["data_path"] = data_path

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


    # Train the model
    ecae_trainer = ECAETrainer(params, device, doc, vae_dir=params.get("vae_dir", None))
    ecae_trainer.train()
    
    # Do the classifier test if required
    
    # Want a higher precision for the test
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    
    # TODO: Maybe pass the samples as file
    # TODO: Could also run over different "modes" here...
    if params.get('do_classifier_test', False):
        print("\n\nstarting classifier test")
        sys.stdout.flush()
        dnn_trainer = DNNTrainer(ecae_trainer, params, device, doc)
        for _ in range(params["classifier_runs"]):
            dnn_trainer.train()
            # TODO: Pass as params parameter
            dnn_trainer.do_classifier_test(do_calibration=True)
            dnn_trainer.reset_model()
        
        # Return the results
        dnn_trainer.clean_up()
        
    # Revert to the previous precision
    torch.set_default_dtype(old_default_dtype)

if __name__=='__main__':
    main()
