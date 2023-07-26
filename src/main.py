import shutil
import argparse
import os

import yaml
import torch

from documenter import Documenter
from trainer import Trainer
from discflow_trainer import *
from disc_class import *
from model import *

def main():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', help='where to find the parameters')
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False,
        help='whether cuda should be used')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
        help='make only plots for traint modle')
    parser.add_argument('-d', '--model_dir', default=None,
        help='model directory for only plot run')
    parser.add_argument('-its', '--model_name')
    args = parser.parse_args()

    with open(args.param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'
    
    if args.plot:
        doc = Documenter(params['run_name'], existing_run=args.model_dir)
    else:
        doc = Documenter(params['run_name'])
    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    print('device: ', device)

    dtype = params.get('dtype', '')
    if dtype=='float64':
        torch.set_default_dtype(torch.float64)
    elif dtype=='float16':
        torch.set_default_dtype(torch.float16)
    elif dtype=='float32':
        torch.set_default_dtype(torch.float32)

    plot_params = {}
    plot_configs = [
        'plot_params/plot_layer_0.yaml',
        'plot_params/plot_layer_1.yaml',
        'plot_params/plot_layer_2.yaml',
        'plot_params/plots.yaml'
    ]
    calo_layer = params.get('calo_layer', None)
    if calo_layer is None:
        for file_name in plot_configs:
            with open(file_name) as f:
                plot_params.update(yaml.load(f, Loader=yaml.FullLoader))
    else:
        with open(plot_configs[calo_layer]) as f:
            plot_params.update(yaml.load(f, Loader=yaml.FullLoader))

    train_loader, test_loader, layer_boundaries = data_util.get_loaders(
            params.get("data_path"),
            params.get("xml_path"),
            params.get("xml_ptype"),
            params.get("val_frac"),
            params.get("batch_size"),
            device=device,
            width_noise=params.get("width_noise"),
            )

    # define disc model
    n_layers = params.get("n_disc_layers", 3)
    n_nodes = params.get("n_disc_nodes", 512)
    disc_dim = params.get("disc_input_dim", 42)
    disc_model = DNN(n_layers, n_nodes, disc_dim)
   
    # define gen model
    gen_model = CINN(params, train_loader.add_noise(train_loader.data), train_loader.cond)

    # define discflow trainer
    trainer = DiscFlow_Trainer(
            params, 
            train_loader, 
            test_loader, 
            layer_boundaries,
            disc_model, 
            gen_model, 
            doc
            )

    if args.plot:
        trainer.load(args.model_name)
    else:
        trainer.train()
    if 'bayesian' in params and params['bayesian']:
        trainer.plot_uncertaintys(plot_params)

if __name__=='__main__':
    main()
