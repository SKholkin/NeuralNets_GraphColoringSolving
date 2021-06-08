import torch
from torch import nn
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from functools import reduce

class GraphNeuralNetworkGCP(nn.Module):
    def __init__(self, max_size, max_n_colors, inner_dim=64, timesteps=32):
        super().__init__()
        self.max_size = max_size
        self.max_n_colors = max_n_colors
        self.timesteps = timesteps
        self.inner_dim = inner_dim
        # very small lstm cells
        # let only color messages to update vertex embeddings
        self.rnn_v = [nn.LSTMCell(input_size=2 * self.inner_dim, hidden_size=self.inner_dim),
                      nn.LSTMCell(input_size=self.inner_dim, hidden_size=self.inner_dim)]
        self.rnn_c = nn.LSTMCell(input_size=self.inner_dim, hidden_size=self.inner_dim)
        self.dropout = nn.Dropout(p=0.3)
        # init Mvv matmul layer (requires_grad=False)
        self.c_msg_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.inner_dim),
            nn.ReLU()
        )
        # init Mvc matmul layers (requires_grad=False)
        self.v_msg_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.inner_dim),
            nn.ReLU()
        )
        self.v_vote_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=self.inner_dim // 4),
            nn.Sigmoid(),
            nn.Linear(in_features=self.inner_dim // 4, out_features=1),
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
        # batch_size, vertex, vetrex_embedding
        vh = [normal.sample(torch.Size([batch_size, self.max_size, self.inner_dim])) for item in self.rnn_v]
        ch = uniform.sample(torch.Size([batch_size, self.max_n_colors, self.inner_dim]))
        v_memory = [torch.zeros(torch.Size([batch_size, self.max_size, self.inner_dim])) for item in self.rnn_v]
        c_memory = torch.zeros(torch.Size([batch_size, self.max_n_colors, self.inner_dim]))
        # run message passing in graph
        for iter in range(self.timesteps):
            muled_by_adj_matr_v = torch.matmul(Mvv, vh[-1])
            color_iter_msg = self.c_msg_mlp(ch)
            vertex_iter_msg = self.v_msg_mlp(vh[-1])
            muled_c_msg = torch.matmul(Mvc, color_iter_msg)
            rnn_vertex_inputs = torch.cat((muled_c_msg, muled_by_adj_matr_v), 2)
            rnn_color_inputs = torch.matmul(torch.transpose(Mvc, 1, 2), vertex_iter_msg)
            
            vh_by_vertex = [[] for item in vh]
            v_memory_by_vertex = [[] for i in v_memory]
            ch_by_vertex = []
            c_memory_by_vertex = []

            for i in range(self.max_size):
                for lstm_num, lstm_cell in enumerate(self.rnn_v):
                    vh_i, v_memory_i = lstm_cell(self.dropout(rnn_vertex_inputs[:,i,:] if lstm_num == 0 else vh[lstm_num - 1][:,i,:]), (vh[lstm_num][:,i,:], v_memory[lstm_num][:,i,:]))
                    vh_by_vertex[lstm_num].append(vh_i)
                    v_memory_by_vertex[lstm_num].append(v_memory_i)

            for i in range(self.max_n_colors):
                ch_i, c_memory_i = self.rnn_c(rnn_color_inputs[:,i,:], (ch[:,i,:], c_memory[:,i,:]))
                ch_by_vertex.append(ch_i)
                c_memory_by_vertex.append(c_memory_i)
            
            # concat
            vh = [reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), vh_by_vertex[lstm_num][1:], vh_by_vertex[lstm_num][0].unsqueeze(1))
             for lstm_num, item in enumerate(vh)]
            v_memory = [reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), v_memory_by_vertex[lstm_num][1:], v_memory_by_vertex[lstm_num][0].unsqueeze(1))
             for lstm_num, item in enumerate(v_memory)]
            ch = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), ch_by_vertex[1:], ch_by_vertex[0].unsqueeze(1))
            c_memory = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), c_memory_by_vertex[1:], c_memory_by_vertex[0].unsqueeze(1))

        #print(f'vertex lstm activations mean {vh[-1].mean(1)}')
        # compute final prediction
        x = self.dropout(vh[-1])
        vote = self.v_vote_mlp(x)
        mean_vote = torch.mean(vote, 1)
        pred = torch.sigmoid(mean_vote).squeeze()
        print(f'pred {torch.var(pred)}')
        return pred
