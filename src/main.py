from functools import partial
import shutil
import argparse
import os
import sys
import yaml
import torch
import os

from documenter import Documenter
from trainer import INNTrainer, DNNTrainer

from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='where to find the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
        help='make only plots for traint model')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'

    # set default parameters for the file locations using the particle type parameter
    dataset = params.get("dataset", 1)
    if "data_path" not in params:
        particle_type = params.get("particle_type", "phoron")
        if dataset == 1:
            data_path = f"/remote/gpu06/ernst/Master_Thesis/Datasets/Dataset1/dataset_1_{particle_type}s_1.hdf5"
        else:
            data_path = f"/remote/gpu06/ernst/Master_Thesis/Datasets/Dataset{dataset}/dataset_{dataset}_1.hdf5"
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

    # Set the default path for the plot configurations used for the uncertainty plots
    plot_params = {}
    plot_configs = [
        os.path.join("..", "plot_params", "plot_layer_0.yaml"),
        os.path.join("..", "plot_params", "plot_layer_1.yaml"),
        os.path.join("..", "plot_params", "plot_layer_2.yaml"),
        os.path.join("..", "plot_params", "plots.yaml")
    ]

    # Load the corresponding plotting file
    calo_layer = params.get('calo_layer', None) # Needed if one only wants to train on one layer
    if calo_layer is None:
        for file_name in plot_configs:
            with open(file_name) as f:
                plot_params.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        with open(plot_configs[calo_layer]) as f:
            plot_params.update(yaml.load(f, Loader=yaml.FullLoader))

    # Parameter used for pretraining.
    # If n_pretrain>0 so many epochs are pretrained with a small fixed log(sigma^2).
    # Possible additional parameters: pretrain_std: & pretrain_max_lr: 
    # Uses the same base LR for both "cycles"
    
    n_pretrain = params.get("n_pretrain", 0)
    
    if n_pretrain > 0:
        
        # Can only be used for a bayesian setup!
        assert params["bayesian"] == True
        pretrain_params = deepcopy(params)
        pretrain_params["n_epochs"] = n_pretrain
        pretrain_params["std_init"] = params.get("pretrain_std", -25)
        pretrain_params["max_lr"] = params.get("pretrain_max_lr", params["max_lr"])

        # Pretrain the model with a fixed std and train the model afterwards normaly with
        # improved initial parameters
        inn_pretrainer = INNTrainer(pretrain_params, device, doc, pretraining=True, plot_params=plot_params)
        inn_pretrainer.train()
        
        inn_trainer = INNTrainer(params, device, doc, plot_params=plot_params)
        inn_trainer.load(update_offset=True)
        inn_trainer.model.reset_sigma()
        inn_trainer.train()
        
    else:
        # Just train the model
        inn_trainer = INNTrainer(params, device, doc, plot_params=plot_params)
        inn_trainer.train()
    
    # Do the classifier test if required
    
    # Want a higher precision for the test
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    
    # TODO: Maybe pass the samples as file
    # TODO: Could also run over different "modes" here...
    if params.get('do_classifier_test', False):
        print("\n\nstarting classifier test")
        sys.stdout.flush()
        dnn_trainer = DNNTrainer(inn_trainer, params, device, doc)
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
