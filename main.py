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

from ColorDataset import ColorDataset
from gnn import GraphNeuralNetworkGCP


def configure_logging(config):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.log_dir = config.log_dir + '/' + current_time
    mkdir(config.log_dir)
    shutil.copy(args.config, config.log_dir + '/' + 'config.json')
    config['tb'] = SummaryWriter(log_dir=config.log_dir)


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--resume', type=str, help='resuming checkpoint')
    parser.add_argument('--data', type=str, help='dataset path')
    parser.add_argument('--save_freq', type=int, help='frequency of saving checkpoints in epochs')
    parser.add_argument('--print_step', type=int, default=100, help='step of printing statistics')
    parser.add_argument('--log_dir', type=str, help='directory for logging and saving checkpoints')
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
    dataset = ColorDataset('datasets')
    criterion = BCELoss()
    model = GraphNeuralNetworkGCP(dataset.max_size, dataset.max_n_colors, timesteps=32)
    if config.resume is not None:
        model = torch.load(config.resume)
    optimizer = Adam(model.parameters(), lr=config.lr)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    train(model, optimizer, config, dataloader, criterion)


def train(model, optimizer, config, dataloader, criterion):
    for epoch in range(config.epochs):
        epoch_len = int(len(dataloader))
        for iter, (Mvv_batch, n_colors_batch, chrom_numb_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(Mvv_batch, n_colors_batch)
            target = tensor([1 if n_colors_batch[batch_elem] >= chrom_numb_batch[batch_elem] else 0
                             for batch_elem in range(n_colors_batch.size(0))],
                            dtype=float32)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if iter % config.print_step == 0:
                print(f'{iter}/{epoch_len} iter Loss: {loss}')
        if epoch % config.save_freq == 0:
            torch.save(model, osp.join(config.log_dir, f'epoch_{epoch}'))


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    config = GNNConfig.from_json(args.config)
    config.update_form_args(args)
    main_worker(config)
