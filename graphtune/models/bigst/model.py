import torch
import torch.nn as nn

from .performer import LinearizedConv


class BigST(nn.Module):
    """
    v2 BigST (legacy-compatible)
    - (Patch) input_emb_layer in_channels should be T_in * in_dim.
      To keep backward-compat when T_in not provided, fallback to T_out.
    """
    def __init__(
        self,
        num_nodes: int,
        T_in: int = None,
        T_out: int = 12,
        in_dim: int = 3,
        hid_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        tau: float = 1.0,
        random_feature_dim: int = 64,
        use_residual: bool = True,
        use_bn: bool = True,
        use_long: bool = False,
        time_num: int = 288,
        week_num: int = 7,
    ):
        super().__init__()
        self.tau = tau
        self.num_layers = num_layers
        self.random_feature_dim = random_feature_dim
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_long = use_long
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.time_num = time_num
        self.week_num = week_num
        self.num_nodes = num_nodes
        self.hid_dim = hid_dim
        self.node_dim = hid_dim
        self.time_dim = hid_dim
        self.output_length = T_out
        self.in_dim = in_dim

        # embeddings
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, self.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, self.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # (Patch) use T_in if given
        if T_in is None:
            T_in = T_out
        self.T_in = int(T_in)
        self.T_out = int(T_out)

        self.input_emb_layer = nn.Conv2d(
            self.T_in * in_dim, hid_dim, kernel_size=(1, 1), bias=True
        )

        # node/time fusion
        self.W_1 = nn.Conv2d(
            self.node_dim + self.time_dim * 2, hid_dim, kernel_size=(1, 1), bias=True
        )
        self.W_2 = nn.Conv2d(
            self.node_dim + self.time_dim * 2, hid_dim, kernel_size=(1, 1), bias=True
        )

        # linearized conv blocks
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linear_conv.append(
                LinearizedConv(hid_dim * 4, hid_dim * 4, dropout, tau, random_feature_dim)
            )
            self.bn.append(nn.LayerNorm(hid_dim * 4))

        # regression head
        if self.use_long:
            self.regression_layer = nn.Conv2d(
                hid_dim * 4 * 2 + hid_dim + T_out, T_out, kernel_size=(1, 1), bias=True
            )
        else:
            self.regression_layer = nn.Conv2d(
                hid_dim * 4 * 2, T_out, kernel_size=(1, 1), bias=True
            )

    def forward(self, x, feat=None):
        # x: (B,N,T_in,3)
        B, N, T, D = x.size()
        time_idx = (x[:, :, -1, 1] * self.time_num).long().clamp(0, self.time_num - 1)
        week_idx = x[:, :, -1, 2].long().clamp(0, self.week_num - 1)

        time_emb = self.time_emb_layer[time_idx.to(self.time_emb_layer.device)]
        week_emb = self.week_emb_layer[week_idx.to(self.week_emb_layer.device)]

        x_flat = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        input_emb = self.input_emb_layer(x_flat)

        node_emb = (
            self.node_emb_layer.unsqueeze(0)
            .expand(B, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)

        x_pool = [x]
        node_vec1 = self.W_1(x_g).permute(0, 2, 3, 1)
        node_vec2 = self.W_2(x_g).permute(0, 2, 3, 1)

        for i in range(self.num_layers):
            residual = x
            x, _, _ = self.linear_conv[i](x, node_vec1, node_vec2)
            if self.use_residual:
                x = x + residual
            if self.use_bn:
                x = self.bn[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)
        x = self.activation(x)

        if self.use_long and feat is not None:
            feat = feat.permute(0, 2, 1).unsqueeze(-1)
            x = torch.cat([x, feat], dim=1)

        x = self.regression_layer(x).squeeze(-1)
        return x, 0.0
