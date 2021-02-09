import torch
from torch import nn
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class GraphNeuralNetworkGCP(nn.Module):
    def __init__(self, max_size, max_n_colors, timesteps=32):
        super().__init__()
        self.max_size = max_size
        self.max_n_colors = max_n_colors
        self.timesteps = timesteps
        # very small lstm cells
        # ToDo: rnn outputs vt sized like inputs (but vt should be d when outputs 2d)
        # let only color messages to update vertex embeddings
        self.rnn_v = nn.LSTMCell(input_size=2 * max_size, hidden_size=max_size)
        self.rnn_c = nn.LSTMCell(input_size=max_n_colors, hidden_size=max_n_colors)
        # init Mvv matmul layer (requires_grad=False)
        self.c_msg_mlp = nn.Sequential(
            nn.Linear(in_features=max_n_colors, out_features=max_n_colors),
            nn.ReLU()
        )
        # init Mvc matmul layers (requires_grad=False)
        self.v_msg_mlp = nn.Sequential(
            nn.Linear(in_features=max_size, out_features=max_size),
            nn.ReLU()
        )
        self.v_vote_mlp = nn.Sequential(
            nn.Linear(in_features=max_size, out_features=max_size),
            nn.Sigmoid()
        )

    def forward(self, Mvv, Mvc):
        batch_size = Mvv.size(0)
        # init base vars
        uniform = Uniform(0, 1)
        normal = Normal(0, 1)
        vh = normal.sample(torch.Size([batch_size, self.max_size]))
        ch = uniform.sample(torch.Size([batch_size, self.max_n_colors]))
        v_memory = torch.zeros(torch.Size([batch_size, self.max_size]))
        c_memory = torch.zeros(torch.Size([batch_size, self.max_n_colors]))
        # run message passing in graph
        for iter in range(self.timesteps):
            color_iter_msg = self.c_msg_mlp(ch)
            vertex_iter_msg = self.v_msg_mlp(vh)
            muled_c_msg = torch.matmul(Mvc, color_iter_msg.unsqueeze(2)).squeeze()
            muled_by_adj_matr_v = torch.matmul(Mvv, vh.unsqueeze(2)).squeeze()
            rnn_vertex_inputs = torch.cat((muled_c_msg, muled_by_adj_matr_v), 1)
            rnn_color_inputs = torch.matmul(torch.transpose(Mvc, 1, 2), vertex_iter_msg.unsqueeze(2)).squeeze()
            vh, v_memory = self.rnn_v(rnn_vertex_inputs, (vh, v_memory))
            ch, c_memory = self.rnn_c(rnn_color_inputs, (ch, c_memory))
        # compute final prediction
        vote = self.v_vote_mlp(vh)
        mean_vote = torch.mean(vote)
        pred = torch.sigmoid(mean_vote)
        return pred
