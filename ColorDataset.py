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
        self.info = torch.load((os.path.join(root,  f'{"train" if is_train else "test"}_info.pt')))
        self.max_size = self.info['nmax']
        self.max_n_colors = self.info['max_n_colors']
        mode = 'train' if is_train else 'test'
        self.basic_data = [torch.load(os.path.join(root, mode, item)) for item in
                               os.listdir(os.path.join(root, mode))]
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
