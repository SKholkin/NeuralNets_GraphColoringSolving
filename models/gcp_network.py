import torch
from torch import nn
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

from functools import reduce

from models.rec_gnn import RecGNN
from models.gcn import GCN

class GraphNeuralNetworkGCP(nn.Module):
    def __init__(self, max_size, max_n_colors, inner_dim=64, timesteps=32, attention=False, attention_version='pairwise_0'):
        super().__init__()
        self.max_size = max_size
        self.max_n_colors = max_n_colors
        self.timesteps = timesteps
        self.inner_dim = inner_dim
        self.rnn_v = 1
        self.v_init = torch.nn.Parameter(Normal(0, 1).sample([self.inner_dim]) / torch.sqrt(torch.Tensor([self.inner_dim])))
        self.preprocess_gnn = GCN(hidden_dim=inner_dim)
        self.rec_gnn = RecGNN(inner_dim, timesteps, attention=attention, attention_version=attention_version)
        self.v_vote_mlp = nn.Sequential(
            nn.Linear(in_features=self.inner_dim, out_features=self.inner_dim // 4),
            nn.Sigmoid(),
            nn.Linear(in_features=self.inner_dim // 4, out_features=1)
        )

    def forward(self, Mvv, n_colors):
        batch_size = Mvv.size(0)
        Mvv += torch.eye(Mvv.size(1)).unsqueeze(0).repeat(batch_size, 1, 1).to(Mvv.device)
        Mvc = torch.tensor([[[1 if j < n_colors[batch_elem] else 0
                              for j in range(self.max_n_colors)]
                             for i in range(self.max_size)]
                            for batch_elem in range(batch_size)], dtype=torch.float32).to(Mvv.device)
        uniform = Uniform(0, 1)
        # batch_size, vertex, vetrex_embedding
        vh = self.v_init.repeat(batch_size, self.max_size, 1).to(Mvv.device)
        ch = uniform.sample(torch.Size([batch_size, self.max_n_colors, self.inner_dim])).to(Mvv.device)

        vh = self.preprocess_gnn(Mvv, vh)
        final_emb = self.rec_gnn(Mvv, Mvc, vh, ch)

        # compute final prediction
        vote = self.v_vote_mlp(final_emb)
        mean_vote = torch.mean(vote, 1)
        pred = torch.sigmoid(mean_vote).squeeze()
        return pred
