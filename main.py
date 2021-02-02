from argparse import ArgumentParser
from ColorDataset import ColorDataset


def main_worker():
    # config parse?
    dataset = ColorDataset('datasets')
    for graph, n_color in dataset:
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main_worker()
