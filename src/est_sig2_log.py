import yaml
import data_util
import math
from trainer import Trainer
from data_util import get_loaders
import torch
import h5py
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mod_dir', help='model directory')
    parser.add_argument('train_data_path')
    parser.add_argument('test_data_path')
    parser.add_argument('gen_data_path')
    parser.add_argument('n_samp', type=int)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('-c', '--use_cuda', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-it', '--iteration')
    args = parser.parse_args()

    with open(args.mod_dir + 'params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = 'cuda:0' if use_cuda else 'cpu'
    print('param file loaded')

    params['train_data_path'] = args.train_data_path
    params['test_data_path'] = args.test_data_path

    doc = None

    trainer = Trainer(params, device, doc)
    states = torch.load(args.mod_dir + 'model' + args.iteration+'.pt', map_location=device)
    trainer.model.load_state_dict(states['net'])
    print('model loaded')

    trainer.model.eval()
    trainer.test_loader.shuffle = False

    gen_loader_1, gen_loader_2 = data_util.get_loaders(
        args.gen_data_path,
        args.gen_data_path,
        params.get('batch_size'),
        params.get('train_split', 0.8),
        device,
        params.get("width_noise", 1e-7),
        params.get("use_extra_dim", False),
        params.get("use_extra_dims", False),
        params.get("mask", 0),
        params.get("calo_layer", None),
        params.get("is_classifier", False)
        )
    
    gen_loader_1.shuffle = False
    gen_loader_2.shuffle = False

    with torch.no_grad():
        logs_res = []
        for i in gen_loader_1:
            logs = []
            for n in range(args.n_samp):
                trainer.model.reset_random()
                log_prob = trainer.model.log_prob(i[0], i[1])
                logs.append(log_prob.detach().cpu())
            logs = torch.vstack(logs).t()
            logs_res.append(logs)
        logs_res = -torch.vstack(logs_res).numpy()
    print(logs_res.shape, logs_res[0])
    if args.append:
        with h5py.File(args.gen_data_path, 'a') as f:
            if 'log_p' not in f.keys():
                f.create_dataset('log_p', data=logs_res.T)
            else:
                del f['log_p']
                f.create_dataset('log_p', data=logs_res.T)
    if args.save:
        np.save('log_sigma_out/log_sig2'+args.iteration+params.get('p_type')+'.npy', logs_res.T)



