import torch
from torch import nn
from functools import reduce
from copy import deepcopy


def masked_softmax(mask, x, dim=2):
    # replace 0 with -1e+06 in mask matrix (exp(1e+06) -> 0)
    mask = deepcopy(mask)
    mask = torch.where(mask == 0.0, torch.tensor(-1e+06, dtype=torch.float32), x)
    return nn.functional.softmax(torch.mul(mask, x), dim=dim)


class RecGNN(nn.Module):
    def __init__(self, inner_dim=64, timesteps=32, attention=False):
        super().__init__()
        self.timesteps = timesteps
        self.inner_dim = inner_dim
        self.rnn_v = [nn.LSTMCell(input_size=2 * self.inner_dim, hidden_size=self.inner_dim)]
        self.rnn_c = nn.LSTMCell(input_size=self.inner_dim, hidden_size=self.inner_dim)
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
        self.attention = attention
        if attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(in_features=2 * self.inner_dim, out_features=1),
                nn.LeakyReLU()
                )

    def forward(self, Mvv, Mvc, vh, ch):
        batch_size = Mvv.size(0)
        max_size = Mvv.size(1)
        max_n_colors = Mvc.size(2)
        v_memory = [torch.zeros(torch.Size([batch_size, max_size, self.inner_dim])) for item in self.rnn_v]
        c_memory = torch.zeros(torch.Size([batch_size, max_n_colors, self.inner_dim]))
        for iter in range(self.timesteps):
            attn_weights = Mvv

            if self.attention:
                concat_tensor_1 = vh[-1].unsqueeze(1).repeat([1, vh[-1].size(1), 1, 1])
                attn_inputs = torch.cat((concat_tensor_1, concat_tensor_1.transpose(1, 2)), dim=3)
                a = self.attn_mlp(attn_inputs).squeeze(3)
                attn_weights = torch.mul(Mvv, a)
                attn_weights = masked_softmax(Mvv, a)
                
            muled_by_adj_matr_v = torch.matmul(attn_weights, vh[-1])
            color_iter_msg = self.c_msg_mlp(ch)
            vertex_iter_msg = self.v_msg_mlp(vh[-1])
            muled_c_msg = torch.matmul(Mvc, color_iter_msg)
            rnn_vertex_inputs = torch.cat((muled_c_msg, muled_by_adj_matr_v), 2)
            rnn_color_inputs = torch.matmul(torch.transpose(Mvc, 1, 2), vertex_iter_msg)
            
            vh_by_vertex = [[] for item in vh]
            v_memory_by_vertex = [[] for i in v_memory]
            ch_by_vertex = []
            c_memory_by_vertex = []

            for i in range(max_size):
                for lstm_num, lstm_cell in enumerate(self.rnn_v):
                    vh_i, v_memory_i = lstm_cell(rnn_vertex_inputs[:,i,:] if lstm_num == 0 else vh[lstm_num - 1][:,i,:],
                                                 (vh[lstm_num][:,i,:], v_memory[lstm_num][:,i,:]))
                    vh_by_vertex[lstm_num].append(vh_i)
                    v_memory_by_vertex[lstm_num].append(v_memory_i)

            for i in range(max_n_colors):
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
        
        return vh[-1]
