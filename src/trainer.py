import sys
import os
import shutil

import torch
import yaml
from tqdm.auto import tqdm

import data
from model import CINN
import plotting

class Trainer:
    def __init__(self, params, device):

        self.params = params
        self.device = device

        train_loader, test_loader = data.get_loaders(
            params.get('data_path'),
            params.get('batch_size'),
            params.get('train_split', 0.8),
            device
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_dim = train_loader.data.shape[1]

        model = CINN(params, device, self.train_loader.data, self.train_loader.cond)
        self.model = model.to(device)
        self.set_optimizer(steps_per_epoch=len(train_loader))

        self.losses_train = []
        self.losses_test = []
        self.learning_rates = []

        os.makedirs(self.get_file(""), exist_ok=True)

    def train(self):

        for epoch in tqdm(range(1,self.params['n_epochs']+1), desc='Epoch',
                         disable=not self.params.get('verbose', True)):
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

            tqdm.write('')
            tqdm.write(f'=== epoch {epoch} ===')
            tqdm.write(f'inn loss (train): {train_loss}')
            tqdm.write(f'inn loss (test): {test_loss}')
            tqdm.write(f'lr: {self.scheduler.get_last_lr()[0]}')
            sys.stdout.flush()

            self.losses_test.append(test_loss)
            plotting.plot_loss(self.get_file('loss.png'), self.losses_train, self.losses_test)
            plotting.plot_lr(self.get_file('learning_rate.png'), self.learning_rates, len(self.train_loader))

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
                    "epoch": self.epoch}, self.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, its optimizer, losses, learning rates and the epoch """
        name = self.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts["net"])

        self.losses_test = state_dicts.get("losses", {})
        self.learning_rates = state_dicts.get("learning_rates", [])
        self.epoch = state_dicts.get("epoch", 0)
        self.optim.load_state_dict(state_dicts["opt"])
        self.model.to(self.device)
    
    def get_file(self, name):
        res_dir = self.params.get('res_dir', 'results')
        return os.path.join(res_dir, name)
    
    def generate(self, num_samples, batch_size = 10000):
        self.model.eval()
        with torch.no_grad():
            energies = 0.99*torch.rand((num_samples,1)) + 0.01
            samples = torch.zeros((num_samples,1,self.num_dim))
            for batch in tqdm(range((num_samples+batch_size-1)//batch_size), desc='Batch',
                         disable=not self.params.get('verbose', True)):
                    start = batch_size*batch
                    stop = min(batch_size*(batch+1), num_samples)
                    energies_l = energies[start:stop].to(self.device)
                    samples[start:stop] = self.model.sample(1, energies_l).cpu()
            samples = samples[:,0,...].cpu().numpy()
            energies = energies.cpu().numpy()
        data.save_data(self.get_file('samples.hdf5'), samples, energies)


def main():
    if len(sys.argv)>=2:
        param_file = sys.argv[1]
    else:
        param_file = 'params/example.yaml'
    with open(param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and not params.get('no_cuda', False)
    device = 'cuda:0' if use_cuda else 'cpu'
    print(device)

    os.makedirs(params['res_dir'], exist_ok=True)
    shutil.copy(param_file, os.path.join(params['res_dir'], 'params.yaml'))

    trainer = Trainer(params, device)
    trainer.train()
    trainer.save()
    trainer.generate(100000)

if __name__=='__main__':
    main()    
