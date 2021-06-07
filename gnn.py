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
        # let only color messages to update vertex embeddings
        self.rnn_v = [nn.LSTMCell(input_size=2 * max_size, hidden_size=max_size),
                      nn.LSTMCell(input_size=max_size, hidden_size=max_size)]
        self.rnn_c = nn.LSTMCell(input_size=max_n_colors, hidden_size=max_n_colors)
        self.dropout = nn.Dropout(p=0.3)
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

    def forward(self, Mvv, n_colors):
        batch_size = Mvv.size(0)
        Mvc = torch.tensor([[[1 if j < n_colors[batch_elem] else 0
                              for j in range(self.max_n_colors)]
                             for i in range(self.max_size)]
                            for batch_elem in range(batch_size)], dtype=torch.float32)
        # init base vars
        uniform = Uniform(0, 1)
        normal = Normal(0, 1)
        vh = [normal.sample(torch.Size([batch_size, self.max_size])) for item in self.rnn_v]
        ch = uniform.sample(torch.Size([batch_size, self.max_n_colors]))
        v_memory = [torch.zeros(torch.Size([batch_size, self.max_size])) for item in self.rnn_v]
        c_memory = torch.zeros(torch.Size([batch_size, self.max_n_colors]))
        # run message passing in graph
        for iter in range(self.timesteps):
            color_iter_msg = self.c_msg_mlp(ch)
            vertex_iter_msg = self.v_msg_mlp(vh[len(self.rnn_v) - 1])
            muled_c_msg = torch.matmul(Mvc, color_iter_msg.unsqueeze(2)).squeeze()
            muled_by_adj_matr_v = torch.matmul(Mvv, vh[len(self.rnn_v) - 1].unsqueeze(2)).squeeze()
            rnn_vertex_inputs = torch.cat((muled_c_msg, muled_by_adj_matr_v), 1)
            rnn_color_inputs = torch.matmul(torch.transpose(Mvc, 1, 2), vertex_iter_msg.unsqueeze(2)).squeeze()
            for i, lstm_cell in enumerate(self.rnn_v):
                vh[i], v_memory[i] = lstm_cell(self.dropout(rnn_vertex_inputs if i == 0 else vh[i - 1]), (vh[i], v_memory[i]))
            ch, c_memory = self.rnn_c(rnn_color_inputs, (ch, c_memory))
        #print(f'vertex lstm activations mean {vh[-1].mean(1)}')
        # compute final prediction
        x = self.dropout(vh[-1])
        vote = self.v_vote_mlp(x)
        mean_vote = torch.mean(vote, 1)
        pred = torch.sigmoid(mean_vote)
        #print(f'pred {pred}')
        return pred
