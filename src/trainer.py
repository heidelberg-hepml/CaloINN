import sys
import shutil
import argparse
import os

import torch
import yaml

import data_util
from model import CINN
import plotting
from documenter import Documenter

class Trainer:
    def __init__(self, params, device, doc):

        self.params = params
        self.device = device
        self.doc = doc

        train_loader, test_loader = data_util.get_loaders(
            params.get('data_path'),
            params.get('batch_size'),
            params.get('train_split', 0.8),
            device,
            params.get("width_noise", 1e-7),
            params.get("use_extra_dim", False),
            params.get("use_extra_dims", False),
            params.get("mask", 0),
            params.get("calo_layer", None)
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_dim = train_loader.data.shape[1]

        data = torch.clone(train_loader.add_noise(train_loader.data))
        cond = torch.clone(train_loader.cond)

        model = CINN(params, device, data, cond)
        self.model = model.to(device)
        self.set_optimizer(steps_per_epoch=len(train_loader))

        self.losses_train = []
        self.losses_test = []
        self.learning_rates = []

    def train(self):

        self.latent_samples(0)

        for epoch in range(1,self.params['n_epochs']+1):
            self.epoch = epoch
            train_loss = 0
            test_loss = 0

            self.model.train()
            for x, c in self.train_loader:
                self.optim.zero_grad()
                loss = - torch.mean(self.model.log_prob(x,c))
                loss.backward()
                self.optim.step()
                self.losses_train.append(loss.detach().cpu().numpy())
                train_loss += self.losses_train[-1]*len(x)
                self.scheduler.step()
                self.learning_rates.append(self.scheduler.get_last_lr()[0])
            train_loss /= len(self.train_loader.data)

            self.model.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    loss = - torch.mean(self.model.log_prob(x,c))
                    test_loss += loss.detach().cpu().numpy()*len(x)
                test_loss /= len(self.test_loader.data)

            print('')
            print(f'=== epoch {epoch} ===')
            print(f'inn loss (train): {train_loss}')
            print(f'inn loss (test): {test_loss}')
            print(f'lr: {self.scheduler.get_last_lr()[0]}')
            sys.stdout.flush()

            self.losses_test.append(test_loss)
            plotting.plot_loss(self.doc.get_file('loss.png'), self.losses_train, self.losses_test)
            plotting.plot_lr(self.doc.get_file('learning_rate.png'), self.learning_rates, len(self.train_loader))

            if epoch%self.params.get("save_interval", 20) == 0 or epoch == self.params['n_epochs']:
                self.save()
                if not epoch == self.params['n_epochs']:
                    self.generate(10000)
                else:
                    self.generate(100000)

                self.latent_samples(epoch)

                plotting.plot_all_hist(
                    self.doc.basedir,
                    self.params['data_path'],
                    mask=self.params.get("mask", 0),
                    calo_layer=self.params.get("calo_layer", None),
                    epoch=epoch)

    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optim = torch.optim.AdamW(
            self.model.params_trainable,
            lr = params.get("lr", 0.0002),
            betas = params.get("betas", [0.9, 0.999]),
            eps = params.get("eps", 1e-6),
            weight_decay = params.get("weight_decay", 0.)
        )

        if no_training: return

        self.lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size = params["lr_decay_epochs"],
                gamma = params["lr_decay_factor"],
            )
        elif self.lr_sched_mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                factor = 0.4,
                patience = 50,
                cooldown = 100,
                threshold = 5e-5,
                threshold_mode = "rel",
                verbose=True
            )
        elif self.lr_sched_mode == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                params.get("max_lr", params["lr"]*10),
                epochs = params.get("opt_epochs") or params["n_epochs"],
                steps_per_epoch=steps_per_epoch)

    def save(self, epoch=""):
        """ Save the model, its optimizer, losses, learning rates and the epoch """
        torch.save({"opt": self.optim.state_dict(),
                    "net": self.model.state_dict(),
                    "losses": self.losses_test,
                    "learning_rates": self.learning_rates,
                    "epoch": self.epoch}, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        self.losses_test = state_dicts.get("losses", {})
        self.learning_rates = state_dicts.get("learning_rates", [])
        self.epoch = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)

    def generate(self, num_samples, batch_size = 10000):
        self.model.eval()
        with torch.no_grad():
            energies = 99.0*torch.rand((num_samples,1)) + 1.0
            samples = torch.zeros((num_samples,1,self.num_dim))
            for batch in range((num_samples+batch_size-1)//batch_size):
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    samples[start:stop] = self.model.sample(1, energies_l).cpu()
            samples = samples[:,0,...].cpu().numpy()
            energies = energies.cpu().numpy()
        samples -= self.params.get("width_noise", 1e-7)
        data_util.save_data(
            self.doc.get_file('samples.hdf5'),
            samples,
            energies,
            use_extra_dim=self.params.get("use_extra_dim", False),
            use_extra_dims=self.params.get("use_extra_dims", False),
            layer=self.params.get("calo_layer", None))

    def latent_samples(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            samples = torch.zeros(self.train_loader.data.shape)
            stop = 0
            for x, c in self.train_loader:
                start = stop
                stop += len(x)
                samples[start:stop] = self.model(x,c)[0].cpu()
            samples = samples.numpy()
        plotting.plot_latent(samples, self.doc.basedir, epoch)


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

    doc = Documenter(params['run_name'])
    shutil.copy(args.param_file, doc.get_file('params.yaml'))
    print('device: ', device)
    print('commit: ', os.popen(r'git rev-parse --short HEAD').read(), end='')

    trainer = Trainer(params, device, doc)
    trainer.train()

if __name__=='__main__':
    main()
