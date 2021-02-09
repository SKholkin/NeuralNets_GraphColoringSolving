from argparse import ArgumentParser
from ColorDataset import ColorDataset
from torch.utils.data import DataLoader
from gnn import GraphNeuralNetworkGCP

def main_worker():
    # config parse?
    dataset = ColorDataset('datasets')
    for graph_data in dataset:
        pass
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    model = GraphNeuralNetworkGCP(dataset.max_size, dataset.max_n_colors, timesteps=2)
    for iter, (Mvv_batch, Mvc_batch) in enumerate(dataloader):
        model(Mvv_batch, Mvc_batch)
    # create dataloader
    # iterate through dataloader


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main_worker()
