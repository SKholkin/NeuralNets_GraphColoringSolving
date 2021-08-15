import torch
from torch.utils.data import Dataset
import os
from utils import adj_list_to_adj_matr


def _transform_to_instance(adj_matr, n_colors, is_solvable, v_size=30, c_size=11):
    # get final matrice instances to put them into GNN
    # Mvv [V,V] adj matr 0 or 1
    # Mvc [V,C] vertex-to-color 1
    # V vertex embeddings from normal
    # C color embeddings (not for each vertex) from uniform
    adj_matr = torch.tensor([[adj_matr[i][j] if (len(adj_matr) > max(j, i)) else 0 for j in range(v_size)]
                             for i in range(v_size)], dtype=torch.float32)
    return adj_matr, n_colors, is_solvable


class ColorDataset(Dataset):

    def __init__(self, root, is_train=True):
        self.root = root
        self.max_size = 30
        self.max_n_colors = 7
        mode = 'train' if is_train else 'test'
        self.basic_data = [torch.load(os.path.join(root, 'ColorDataset', mode, item)) for item in
                               os.listdir(os.path.join(root, 'ColorDataset', mode))]
        self.data = []
        for graph_info in self.basic_data:
            self.data.append((graph_info['adj_list'], graph_info['n_colors'], graph_info['is_solvable']))

    def __getitem__(self, idx):
        # get instance through transformation
        adj_matr = adj_list_to_adj_matr(self.data[idx][0])
        n_color = self.data[idx][1]
        is_solvable = self.data[idx][2]
        return _transform_to_instance(adj_matr, n_color, is_solvable, v_size=self.max_size, c_size=self.max_n_colors)

    def __len__(self):
        return len(self.basic_data)


def prepare_folders(root, clear_up=False, is_test=False):
    mode = 'test' if is_test else 'train'
    if not os.path.exists(root):
        raise RuntimeError('root dataset path do not exist')
    if not os.path.exists(os.path.join(root, 'ColorDataset')):
        os.mkdir(os.path.join(root, 'ColorDataset'))
    if not os.path.exists(os.path.join(root, 'ColorDataset', mode)):
        os.mkdir(os.path.join(root, 'ColorDataset', mode))
    else:
        if clear_up:
            [os.remove(os.path.join(root, 'ColorDataset', mode, f)) for f in os.listdir(
                os.path.join(root, 'ColorDataset', mode))]