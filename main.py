from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch import tensor, float32
from torch.utils.tensorboard import SummaryWriter
import json
from addict import Dict
from os import path as osp
from os import mkdir
import datetime
import shutil
from statistics import mean

from ColorDataset import ColorDataset
from gnn import GraphNeuralNetworkGCP
from utils import AverageMetr


def configure_logging(config):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.log_dir = config.log_dir + '/' + current_time
    mkdir(config.log_dir)
    shutil.copy(args.config, config.log_dir + '/' + 'config.json')
    config['tb'] = SummaryWriter(log_dir=config.log_dir)


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--resume', type=str, help='resuming checkpoint')
    parser.add_argument('--data', type=str, help='dataset path')
    parser.add_argument('--save_freq', type=int, help='frequency of saving checkpoints in epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='step of printing statistics')
    parser.add_argument('--log_dir', type=str, help='directory for logging and saving checkpoints')
    parser.add_argument('--test_freq', type=int, default=5, help='test every (test_freq) epoch')
    return parser


class GNNConfig(Dict):
    @classmethod
    def from_json(self, path) -> 'GNNConfig':
        with open(path) as f:
            loaded_json = json.load(f)
        return self(loaded_json)

    def update_form_args(self, args):
        for key, value in vars(args).items():
            if key in self.keys():
                if value is not None:
                    self[key] = value
                else:
                    continue
            self[key] = value


def main_worker(config):
    # config parse?
    configure_logging(config)
    train_dataset = ColorDataset('datasets', is_train=True)
    val_dataset = ColorDataset('datasets', is_train=False)
    criterion = BCELoss()
    model = GraphNeuralNetworkGCP(train_dataset.max_size, train_dataset.max_n_colors, timesteps=32)
    if config.resume is not None:
        model = torch.load(config.resume)
    optimizer = Adam(model.parameters(), lr=config.lr)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    if config.mode == 'train':
        train(model, optimizer, config, train_loader, val_loader, criterion)
    else:
        validate(model, optimizer, config, val_loader, config, 0)


# ToDo: calculate accuracy
def train(model, optimizer, config, train_loader, val_loader, criterion):
    best_acc = 0
    for epoch in range(config.epochs):
        print(f'Training epoch {epoch}')
        avg_loss = AverageMetr()
        avg_acc = AverageMetr()
        epoch_len = int(len(train_loader))
        for iter, (Mvv_batch, n_colors_batch, chrom_numb_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(Mvv_batch, n_colors_batch)
            target = tensor([1 if n_colors_batch[batch_elem] >= chrom_numb_batch[batch_elem] else 0
                             for batch_elem in range(n_colors_batch.size(0))],
                            dtype=float32)
            loss = criterion(output, target)
            acc = compute_acc(output, target)
            loss.backward()
            optimizer.step()

            avg_loss.update(loss)
            avg_acc.update(acc)
            if iter % config.print_freq == 0:
                print(f'{iter}/{epoch_len} iter Loss: {loss}({avg_loss.avg()}) Acc: {acc}')
        config.tb.add_scalar('Train/Loss', avg_loss.avg(), epoch)
        config.tb.add_scalar('Train/Acc', avg_acc.avg(), epoch)
        print(f'Average epoch {epoch}: loss {avg_loss.avg()}:acc {avg_acc.avg()}')
        if epoch % config.save_freq == 0:
            torch.save(model, osp.join(config.log_dir, f'epoch_{epoch}'))
            print('Successfully saved checkpoint')
        if epoch % config.test_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, config, epoch)
            best_acc = max(best_acc, val_acc)


def validate(model, val_loader, criterion, config, epoch):
    avg_loss = AverageMetr()
    avg_acc = AverageMetr()
    print('Validating...')
    with torch.no_grad():
        for iter, (Mvv_batch, n_colors_batch, chrom_numb_batch) in enumerate(val_loader):
            output = model(Mvv_batch, n_colors_batch)
            target = tensor([1 if n_colors_batch[batch_elem] >= chrom_numb_batch[batch_elem] else 0
                             for batch_elem in range(n_colors_batch.size(0))],
                             dtype=float32)
            loss = criterion(output, target)
            acc = compute_acc(output, target)
            avg_loss.update(loss)
            avg_acc.update(acc)
            if iter % config.print_freq == 0:
                print(f'iter {iter} Loss {loss} Acc {acc}')
    config.tb.add_scalar('Val/Loss', avg_loss.avg(), epoch)
    config.tb.add_scalar('Val/Acc', avg_acc.avg(), epoch)
    print(f'Average Loss {avg_loss.avg()} Acc {avg_acc.avg()}\n')
    return avg_loss.avg(), avg_acc.avg()


def compute_acc(output, target):
    return mean([1 if abs(output[i] - target[i]) < 0.5 else 0 for i, x in enumerate(output)])


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    config = GNNConfig.from_json(args.config)
    config.update_form_args(args)
    main_worker(config)
