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
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)

    def forward(self, a, x, sym_norm_laplacian=None):
        if sym_norm_laplacian is None:
            sym_norm_laplacian = calculate_sym_normalized_Laplacian(a)
        x = torch.matmul(sym_norm_laplacian, x)
        x = self.linear(x)
        return self.relu(x)


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
    def __init__(self, max_v, max_n_color, dim_per_color = 10):
        super().__init__()
        self.dim_per_color = dim_per_color
        self.max_n_color = max_n_color
        self.max_v = max_v
        self.gcn_v1 = GCNLayer(input_dim=self.max_n_color * self.dim_per_color, hidden_dim=self.max_n_color * self.dim_per_color)
        self.gcn_v2 = GCNLayer(input_dim=self.max_n_color * self.dim_per_color, hidden_dim=self.max_n_color * self.dim_per_color)
        self.uniform = Uniform(0, 1)
        self.v_vote_mlp = nn.Sequential(
            nn.Linear(in_features=self.max_n_color * self.dim_per_color, out_features=self.max_n_color * self.dim_per_color // 4),
            nn.Sigmoid(),
            nn.Linear(in_features=self.max_n_color * self.dim_per_color // 4, out_features=1)
        )

    def forward(self, a, n_colors):
        batch_size = a.size(0)
        color_mask = torch.tensor([[[1 if j < n_colors[batch_elem] * self.dim_per_color else 0
                              for j in range(self.max_n_color * self.dim_per_color)]
                             for i in range(self.max_v)]
                            for batch_elem in range(batch_size)], dtype=torch.float32, requires_grad=False)
        a += torch.eye(a.size(1))

        sym_norm_laplacian = calculate_sym_normalized_Laplacian(a)
        v_features = self.uniform.sample(torch.Size([batch_size, a.size(1), self.max_n_color * self.dim_per_color]))
        v_features = self.gcn_v1(a, torch.mul(v_features, color_mask), sym_norm_laplacian)
        v_features = self.gcn_v2(a, torch.mul(v_features, color_mask), sym_norm_laplacian)

        vote = self.v_vote_mlp(v_features)
        mean_vote = torch.mean(vote, 1)
        pred = torch.sigmoid(mean_vote).squeeze()
        return pred
