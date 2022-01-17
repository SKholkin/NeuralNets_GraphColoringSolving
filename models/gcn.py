import torch
from torch import nn
from torch.autograd.grad_mode import F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


def calculate_sym_normalized_Laplacian(A):
    degree_vector = torch.pow(torch.sum(A, dim=1), -1/2)
    degree_matr = torch.diag_embed(degree_vector, dim1=-2, dim2=-1)
    return torch.bmm(torch.bmm(degree_matr, A), degree_matr)


class GCNLayer(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, Mvv, x):
        x = torch.matmul(Mvv, x)
        x = self.linear(x)
        x = self.relu(x)
        return x


class GCNTyped(nn.Module):
    def __init__(self, max_v, max_n_colors):
        super().__init__()
        self.hidden_dim_list = [64, 64 ,64]
        self.max_n_colors = max_n_colors
        self.max_v = max_v
        self.gcn_v1 = GCNLayer(input_dim=2 * self.hidden_dim_list[0], hidden_dim=self.hidden_dim_list[1])
        self.gcn_v2 = GCNLayer(input_dim=2 * self.hidden_dim_list[1], hidden_dim=self.self.hidden_dim_list[2])
        self.c_msg_2 = nn.Linear(in_features=self.hidden_dim_list[1], out_features=self.hidden_dim_list[1])
        self.c_update_1 = nn.Sequential(nn.Linear(in_features=2 * self.hidden_dim_list[0], out_features=self.hidden_dim_list[1]), nn.ReLU())
        self.normal = Normal(0, 1)

    def forward(self, a, n_color):
        # peace of shit honestly
        # DOESN'T CONTROL COLOR !!!!!
        batch_size = a.size(0)
        sym_norm_laplacian = calculate_sym_normalized_Laplacian(a)
        v_features = self.normal.sample(torch.Size([batch_size, self.max_v, self.hidden_dim_list[0]]))
        c_features = self.normal.sample(torch.Size([batch_size, self.max_n_colors, self.hidden_dim_list[0]]))
        v_features_0 = torch.cat(v_features, torch.sum(c_features, 1), 2)
        c_features_0 = torch.cat(c_features, torch.sum(v_features, 1), 2)
        v_features = self.gcn_v1(a, v_features_0, sym_norm_laplacian)
        c_features = self.c_update_1(c_features_0)
        c_features_for_v = self.c_msg_2(c_features)
        v_features_0 = torch.cat(v_features, torch.sum(c_features_for_v, 1), 2)
        v_features = self.gcn_v2(a, v_features_0, sym_norm_laplacian)
        return v_features


class GCN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gcn_v1 = GCNLayer(hidden_dim=self.hidden_dim)
        self.gcn_v2 = GCNLayer(hidden_dim=self.hidden_dim)
        self.gcn_v3 = GCNLayer(hidden_dim=self.hidden_dim)

    def forward(self, Mvv, x):
        x = self.gcn_v1(Mvv, x)
        x = self.gcn_v2(Mvv, x)
        x = self.gcn_v3(Mvv, x)
        return x
