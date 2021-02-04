from argparse import ArgumentParser
from ColorDataset import ColorDataset
from torch.utils.data import DataLoader


def main_worker():
    # config parse?
    dataset = ColorDataset('datasets')
    for graph_data in dataset:
        pass
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for iter, batch in enumerate(dataloader):
        pass
    # create dataloader
    # iterate through dataloader
    #


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main_worker()
