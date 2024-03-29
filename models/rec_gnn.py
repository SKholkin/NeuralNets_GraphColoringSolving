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
    def __init__(self, inner_dim=64, timesteps=32, attention=False, attention_version='pairwise_0'):
        super().__init__()
        self.timesteps = timesteps
        self.inner_dim = inner_dim
        self.dropout = nn.Dropout(p=0.3)
        
        self.rnn_v = nn.LSTMCell(input_size=2 * self.inner_dim, hidden_size=self.inner_dim)
        self.rnn_c = nn.LSTMCell(input_size=self.inner_dim, hidden_size=self.inner_dim)
        self.c_msg_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.inner_dim),
            nn.ReLU()
        )
        self.v_msg_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.inner_dim),
            nn.ReLU()
        )
        self.attention = attention
        self.attention_version = attention_version if attention else None
        print(f'Attention version {self.attention_version} Inner dim {self.inner_dim}')
        if attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(in_features=2 * self.inner_dim, out_features=self.inner_dim // 2),
                nn.Sigmoid(),
                nn.Linear(in_features=self.inner_dim // 2, out_features=1),
                nn.LeakyReLU()
                )
            if self.attention_version == 'pairwise_2' or self.attention_version == 'pairwise_3':
                self.attn_inputs_mlp1 = nn.Linear(in_features=2 * self.inner_dim, out_features=2 * self.inner_dim)
        self.vh_mlp = nn.Linear(in_features=self.inner_dim, out_features=self.inner_dim)

    def forward(self, Mvv, Mvc, vh, ch):
        batch_size = Mvv.size(0)
        max_size = Mvv.size(1)
        max_n_colors = Mvc.size(2)
        v_memory = torch.zeros(torch.Size([batch_size, max_size, self.inner_dim])).to(Mvv.device)
        c_memory = torch.zeros(torch.Size([batch_size, max_n_colors, self.inner_dim])).to(Mvv.device)
        for iter in range(self.timesteps):
            attn_weights = Mvv

            vh_signal = vh
            if self.attention_version == 'pairwise_1' or self.attention_version == 'pairwise_2' or self.attention_version == 'pairwise_3':
                vh_signal = self.vh_mlp(vh_signal)

            if self.attention:
                concat_tensor_1 = vh_signal.unsqueeze(1).repeat([1, vh_signal.size(1), 1, 1])
                attn_inputs = torch.cat((concat_tensor_1, concat_tensor_1.transpose(1, 2)), dim=3)
                a = self.attn_mlp(attn_inputs).squeeze(3)
                attn_weights = torch.mul(Mvv, a)
            
            muled_by_adj_matr_v = torch.matmul(attn_weights, vh_signal)
            color_iter_msg = self.c_msg_mlp(ch)
            vertex_iter_msg = self.v_msg_mlp(vh[-1])
            muled_c_msg = torch.matmul(Mvc, color_iter_msg)
            rnn_vertex_inputs = torch.cat((muled_c_msg, muled_by_adj_matr_v), 2)
            rnn_color_inputs = torch.matmul(torch.transpose(Mvc, 1, 2), vertex_iter_msg)
            
            vh_by_vertex = []
            v_memory_by_vertex = []
            ch_by_vertex = []
            c_memory_by_vertex = []

            # for i in range(max_size):
            #     for lstm_num, lstm_cell in enumerate(self.rnn_v):
            #         vh_i, v_memory_i = lstm_cell(rnn_vertex_inputs[:,i,:] if lstm_num == 0 else vh[lstm_num - 1][:,i,:],
            #                                      (vh[lstm_num][:,i,:], v_memory[lstm_num][:,i,:]))
            #         vh_by_vertex[lstm_num].append(vh_i)
            #         v_memory_by_vertex[lstm_num].append(v_memory_i)

            for i in range(max_size):
                vh_i, v_memory_i = self.rnn_v(rnn_vertex_inputs[:,i,:], (vh[:,i,:], v_memory[:,i,:]))
                vh_by_vertex.append(vh_i)
                v_memory_by_vertex.append(v_memory_i)

            for i in range(max_n_colors):
                ch_i, c_memory_i = self.rnn_c(rnn_color_inputs[:,i,:], (ch[:,i,:], c_memory[:,i,:]))
                ch_by_vertex.append(ch_i)
                c_memory_by_vertex.append(c_memory_i)
            
            # concat
            # vh = [reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), vh_by_vertex[lstm_num][1:], vh_by_vertex[lstm_num][0].unsqueeze(1))
            #  for lstm_num, item in enumerate(vh)]
            # v_memory = [reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), v_memory_by_vertex[lstm_num][1:], v_memory_by_vertex[lstm_num][0].unsqueeze(1))
            #  for lstm_num, item in enumerate(v_memory)]
             
            vh = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), vh_by_vertex[1:], vh_by_vertex[0].unsqueeze(1))
            v_memory = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), v_memory_by_vertex[1:], v_memory_by_vertex[0].unsqueeze(1))
            ch = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), ch_by_vertex[1:], ch_by_vertex[0].unsqueeze(1))
            c_memory = reduce(lambda a, b: torch.cat((a, b.unsqueeze(1)), dim=1), c_memory_by_vertex[1:], c_memory_by_vertex[0].unsqueeze(1))
        
        return vh
