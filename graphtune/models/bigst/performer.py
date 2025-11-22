import math
import torch
import torch.nn as nn

from .layers import create_random_matrix, random_feature_map, linear_kernel


class ConvApproximation(nn.Module):
    def __init__(self, dropout: float, tau: float, random_feature_dim: int):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        dim = node_vec1.shape[-1]
        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(
            self.random_feature_dim, dim, seed=int(random_seed.item())
        ).to(node_vec1.device)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)
        return x, node_vec1_prime, node_vec2_prime


class LinearizedConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        dropout: float,
        tau: float = 1.0,
        random_feature_dim: int = 64,
    ):
        super().__init__()
        # attribute names 유지 (state_dict 호환)
        self.input_fc = nn.Conv2d(
            in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True
        )
        self.output_fc = nn.Conv2d(
            in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True
        )
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_app_layer = ConvApproximation(dropout, tau, random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)
        x = x.permute(0, 2, 3, 1)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)
        return x, node_vec1_prime, node_vec2_prime
