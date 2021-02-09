import torch
from torch.utils.data import Dataset
import os
from utils import adj_list_to_adj_matr
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


def _transform_to_instance(adj_matr, n_colors, chromatic_numb, v_size=30, c_size=11):
    # get final matrice instances to put them into GNN
    # Mvv [V,V] adj matr 0 or 1
    # Mvc [V,C] vertex-to-color 1
    # V vertex embeddings from normal
    # C color embeddings (not for each vertex) from uniform
    adj_matr = torch.tensor([[adj_matr[i][j] if (len(adj_matr) > max(i, j)) else 0
                              for j in range(v_size)] for i in range(v_size)], dtype=torch.float32)
    return adj_matr, n_colors, chromatic_numb


class ColorDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.max_size = 30
        self.max_n_colors = 11
        self.adv_graph_data = [torch.load(os.path.join(root, 'ColorDataset', 'adv', f)) for f in
                               os.listdir(os.path.join(root, 'ColorDataset', 'adv'))]
        self.basic_graph_data = [torch.load(os.path.join(root, 'ColorDataset', 'basic', f)) for f in
                                 os.listdir(os.path.join(root, 'ColorDataset', 'basic'))]
        # in data graphs encoded through adj lists
        self.basic_data = []
        for i in range(len(self.basic_graph_data)):
            self.basic_data.append(self.basic_graph_data[i])
            if i < len(self.adv_graph_data):
                self.basic_data.append(self.adv_graph_data[i])
        self.data = []
        for graph_info in self.basic_data:
            self.data += [(graph_info[1:], n_color, graph_info[0]) for n_color
                          in range(max(2, graph_info[0] - 2), graph_info[0] + 3)]

    def __getitem__(self, idx):
        # get instance through transformation
        adj_matr = adj_list_to_adj_matr(self.data[idx][0])
        n_color = self.data[idx][1]
        chromatic_numb = self.data[idx][2]
        return _transform_to_instance(adj_matr, n_color, chromatic_numb, v_size=self.max_size, c_size=self.max_n_colors)

    def __len__(self):
        return len(self.basic_data)


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