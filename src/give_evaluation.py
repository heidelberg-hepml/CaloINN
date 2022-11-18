from functools import partial
import shutil
import argparse
import os
import sys
import yaml
import torch

from documenter import Documenter
from trainer import INNTrainer, DNNTrainer
import plotting

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

    # set default parameters for the file locations
    if "data_path_train" not in params:
        particle = params.get("particle_type", "piplus")
        data_path_train = os.path.join("..", "Datasets", particle , "train_" + particle + ".hdf5")
        params["data_path_train"] = data_path_train
        
    if "data_path_test" not in params:
        particle = params.get("particle_type", "piplus")
        data_path_test = os.path.join("..", "Datasets", particle , "test_" + particle + ".hdf5")
        params["data_path_test"] = data_path_test

    if "classification_set" not in params:
        particle = params.get("particle_type", "piplus")
        classification_set = os.path.join("..", "Datasets", particle , "cls_" + particle + ".hdf5")
        params["classification_set"] = classification_set


    doc = Documenter(params['run_name'], existing_run=True)
    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    print('device: ', device)
    print('commit: ', os.popen(r'git rev-parse --short HEAD').read(), end='')

    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

    plot_params = {}
    # TODO: Use os.path.join!
    plot_configs = [
        '../plot_params/plot_layer_0.yaml',
        '../plot_params/plot_layer_1.yaml',
        '../plot_params/plot_layer_2.yaml',
        '../plot_params/plots.yaml'
    ]
    calo_layer = params.get('calo_layer', None)
    if calo_layer is None:
        for file_name in plot_configs:
            with open(file_name) as f:
                plot_params.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        with open(plot_configs[calo_layer]) as f:
            plot_params.update(yaml.load(f, Loader=yaml.FullLoader))

    
    inn_trainer = INNTrainer(params, device, doc)
    inn_trainer.load()
    inn_trainer.generate(100000)
    
    plotting.plot_all_hist(
                    doc.basedir,
                    params['data_path_test'],
                    mask=params.get("mask", 0),
                    calo_layer=params.get("calo_layer", None),
                    epoch=inn_trainer.epoch,
                    p_ref=params.get("particle_type", "piplus"),
                    in_one_file=True)

    print("starting classifier test")
    sys.stdout.flush()
    if params.get('do_classifier_test', False):
        dnn_trainer = DNNTrainer(params, device, doc)
        for _ in range(params["classifier_runs"]):
            dnn_trainer.train()
            # TODO: Pass as params parameter
            dnn_trainer.do_classifier_test(do_calibration=True)
            dnn_trainer.reset_model()
            
        # Needed for final table and to fix dtype bug!
        dnn_trainer.clean_up()
    
    if 'bayesian' in params and params['bayesian']:
        inn_trainer.plot_uncertaintys(plot_params)

if __name__=='__main__':
    main()
