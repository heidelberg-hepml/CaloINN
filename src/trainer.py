import sys

import torch

import yaml
from tqdm.auto import tqdm

from data import get_loaders
from model import CINN

class Trainer:
    def __init__(self, params, device):

        self.params = params
        self.device = device

        train_loader, test_loader = get_loaders(
            params.get('data_path'),
            params.get('batch_size'),
            params.get('train_split', 0.8),
            device
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_dim = train_loader.data.shape[1]

    def train(self, inn):

        losses_train = []
        losses_test = []

        inn.initialize_normalization(self.train_loader.data)
        inn.define_model_architecture(self.num_dim)
        inn.set_optimizer(steps_per_epoch=len(self.train_loader))

        for epoch in tqdm(range(1,self.params['n_epochs']+1), desc='Epoch', disable=not self.params.get('verbose', True)):
            train_loss = 0
            test_loss = 0

            inn.train()
            for x, c in self.train_loader:
                inn.optim.zero_grad()
                loss = - torch.mean(inn.log_prob(x,c))
                loss.backward()
                inn.optim.step()
                train_loss += loss.detach().cpu().numpy()*len(x)
                inn.scheduler.step()
            train_loss /= len(self.train_loader.data)

            inn.eval()
            with torch.no_grad():
                for x, c in self.test_loader:
                    loss = - torch.mean(inn.log_prob(x,c))
                    test_loss += loss.detach().cpu().numpy()*len(x)
                test_loss /= len(self.test_loader.data)

                tqdm.write('')
                tqdm.write(f'=== epoch {epoch} ===')
                tqdm.write(f'inn loss (train): {train_loss}')
                tqdm.write(f'inn loss (test): {test_loss}')
                tqdm.write(f'lr: {inn.scheduler.get_last_lr()[0]}')
                sys.stdout.flush()
            
            losses_train.append(train_loss)
            losses_test.append(test_loss)

        # if result_dir:
        #     np.save(os.path.join(result_dir, 'losses_over_epochs_train.npy'), np.array(losses_over_epochs_train))
        #     np.save(os.path.join(result_dir, 'losses_over_epochs_test.npy'), np.array(losses_over_epochs_test))

def main():
    if len(sys.argv)>=3:
        param_file = sys.argv[-1]
    else:
        param_file = 'params/example.yaml'
    with open(param_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    use_cuda = torch.cuda.is_available() and not params.get('no_cuda', False)
    device = 'cuda:0' if use_cuda else 'cpu'
    print(device)
    trainer = Trainer(params, device)
    inn = CINN(params, device)
    trainer.train(inn)

if __name__=='__main__':
    main()    
