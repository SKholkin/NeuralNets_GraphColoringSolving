import torch
from torch.utils.data import Dataset
import os
import random
from utils import adj_list_to_adj_matr


class ColorDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.adv_graph_data = [torch.load(os.path.join(root, 'ColorDataset', 'adv', f)) for f in
                               os.listdir(os.path.join(root, 'ColorDataset', 'adv'))]
        self.basic_graph_data = [torch.load(os.path.join(root, 'ColorDataset', 'basic', f)) for f in
                                 os.listdir(os.path.join(root, 'ColorDataset', 'basic'))]
        # in data graphs encoded through adj lists
        self.data = []
        for i in range(len(self.basic_graph_data)):
            self.data.append(self.basic_graph_data[i])
            if i < len(self.adv_graph_data):
                self.data.append(self.adv_graph_data[i])
        #self.data = random.shuffle(self.adv_graph_data + self.basic_graph_data)
        pass

    def __getitem__(self, idx):
        # get graph
        # get instance through transformation
        n_color = self.data[idx][0]
        adj_matr = adj_list_to_adj_matr(self.data[idx][1:])
        return adj_matr, n_color

    def __len__(self):
        return len(self.data)


class TransformToInstance:
    def __init__(self):
        pass

    def __call__(self, graph, n_colors):
        # get final matrice instances to put them into GNN
        matrices = None
        return matrices


def prepare_folders(root, clear_up=False):
    if not os.path.exists(root):
        raise RuntimeError('root dataset path do not exist')
    if not os.path.exists(os.path.join(root, 'ColorDataset')):
        os.mkdir(os.path.join(root, 'ColorDataset'))
    if not os.path.exists(os.path.join(root, 'ColorDataset', 'basic')):
        os.mkdir(os.path.join(root, 'ColorDataset', 'basic'))
    else:
        if clear_up:
            [os.remove(os.path.join(root, 'ColorDataset', 'basic', f) for f in os.listdir(
                os.path.join(root, 'ColorDataset', 'basic')))]
    if not os.path.exists(os.path.join(root, 'ColorDataset', 'adv')):
        os.mkdir(os.path.join(root, 'ColorDataset', 'adv'))
    else:
        if clear_up:
            [os.remove(os.path.join(root, 'ColorDataset', 'adv', f) for f in os.listdir(
                os.path.join(root, 'ColorDataset', 'adv')))]