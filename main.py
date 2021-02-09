from argparse import ArgumentParser
from ColorDataset import ColorDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch import tensor, float32
from gnn import GraphNeuralNetworkGCP


def get_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, help='dataset path')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--save_freq', type=int, help='frequency of saving checkpoints in epochs')
    parser.add_argument('--lr', type=float)
    return parser


class MyConfig:
    def __init__(self):
        pass


def main_worker(config):
    # config parse?
    dataset = ColorDataset('datasets')
    criterion = BCELoss()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    model = GraphNeuralNetworkGCP(dataset.max_size, dataset.max_n_colors, timesteps=2)
    optimizer = Adam(model.parameters(), lr=config.lr)
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
            if iter % 10 == 0:
                print(f'{iter}/{epoch_len} iter Loss: {loss}')


if __name__ == '__main__':
    parser = get_argument_parser()
    config = parser.parse_args()
    main_worker(config)
