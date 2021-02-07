from torch import nn


class GraphNeuralNetworkGCP(nn.Module):
    def __init__(self, max_size, max_n_colors, timesteps=32):
        super().__init__()
        self.max_size = max_size
        self.max_n_colors = max_n_colors
        self.timesteps = timesteps
        # very small lstm cells
        # ToDo: rnn outputs vt sized like inputs (but vt should be d when outputs 2d)
        self.rnn_v = nn.LSTMCell(input_size=2 * max_size, hidden_size=max_size, bias=True)
        self.rnn_c = nn.LSTMCell(input_size=max_n_colors, hidden_size=max_n_colors, bias=True)
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

    def forward(self, Mvv, Mvc, V, C):
        pass
