from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import BCELoss
from torch.utils.tensorboard import SummaryWriter
import json
from addict import Dict
from os import path as osp
from os import mkdir
import datetime
import shutil
from statistics import mean
from ColorDataset import ColorDataset
from models.gcp_network import GraphNeuralNetworkGCP
from models.gcn import GCN
from utils import AverageMetr


def configure_logging(config):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if config.log_dir is None:
        config.log_dir = 'log_dir'
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
    parser.add_argument('--save_freq', type=int, help='frequency of saving checkpoints in epochs', default=5)
    parser.add_argument('--print_freq', type=int, help='step of printing statistics', default=10)
    parser.add_argument('--log_dir', type=str, help='directory for logging and saving checkpoints', default='log_dir')
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
    configure_logging(config)
    train_dataset = ColorDataset('datasets', is_train=True)
    val_dataset = ColorDataset('datasets', is_train=False)
    max_size = max(train_dataset.max_size, val_dataset.max_size)
    max_n_colors = max(train_dataset.max_n_colors, val_dataset.max_n_colors)

    criterion = BCELoss()
    model = GraphNeuralNetworkGCP(max_size, max_n_colors, timesteps=config.timesteps, attention=config.attention, attention_version=config.attention_version, inner_dim=64)
    if config.resume is not None:
        model = torch.load(config.resume)
    
    print(f'Total number of parameters of model: {sum([item.numel() for item in model.parameters()])}')
    optimizer = Adam(model.parameters(), lr=config.lr)
    lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_steps)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    if config.mode == 'train':
        train(model, optimizer, lr_scheduler, config, train_loader, val_loader, criterion)
    else:
        validate(model, val_loader, criterion, config, 0)


def train(model, optimizer, lr_scheduler, config, train_loader, val_loader, criterion):
    best_acc = 0
    for epoch in range(config.epochs):
        print(f'Training epoch {epoch}')
        avg_loss = AverageMetr()
        avg_acc = AverageMetr()
        epoch_len = int(len(train_loader))
        for iter, (Mvv_batch, n_colors_batch, is_solvable_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(Mvv_batch, n_colors_batch)
            loss = criterion(output, is_solvable_batch.float())
            acc = compute_acc(output, is_solvable_batch.float())
            loss.backward()

            optimizer.step()

            avg_loss.update(float(loss))
            avg_acc.update(float(acc))
            if iter % config.print_freq == 0:
                print(f'{iter}/{epoch_len} iter Loss: {loss}({avg_loss.avg()}) Acc: {acc} Lr: {lr_scheduler.get_last_lr()}')
            if (iter + 1) % 100 == 0:
                break
        
        lr_scheduler.step()
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
        for iter, (Mvv_batch, n_colors_batch, is_solvable_batch) in enumerate(val_loader):
            output = model(Mvv_batch, n_colors_batch)
            loss = criterion(output, is_solvable_batch.float())
            acc = compute_acc(output, is_solvable_batch.float())
            avg_loss.update(float(loss))
            avg_acc.update(float(acc))
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
