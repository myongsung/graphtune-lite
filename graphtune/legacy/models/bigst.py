import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def create_random_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])

    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}.")

    return torch.matmul(torch.diag(multiplier), final_matrix)

def random_feature_map(data, is_query, projection_matrix, numerical_stabilizer=1e-6):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32, device=data.device)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32, device=data.device))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1) / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)

    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3

    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer)
    else:
        data_dash = ratio * (torch.exp(
            data_dash - diag_data -
            torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0], dim=attention_dims_t, keepdim=True)[0]
        ) + numerical_stabilizer)
    return data_dash

def linear_kernel(x, node_vec1, node_vec2):
    node_vec1 = node_vec1.permute(1, 0, 2, 3)
    node_vec2 = node_vec2.permute(1, 0, 2, 3)
    x = x.permute(1, 0, 2, 3)

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)

    one_matrix = torch.ones([node_vec2.shape[0]], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    out1 = out1.permute(1, 0, 2, 3)
    out2 = out2.permute(1, 0, 2).unsqueeze(-1)
    return out1 / out2

class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        dim = node_vec1.shape[-1]
        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed=int(random_seed.item())).to(node_vec1.device)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)
        return x, node_vec1_prime, node_vec2_prime

class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super().__init__()
        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_app_layer = conv_approximation(dropout, tau, random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)
        x = x.permute(0, 2, 3, 1)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)
        return x, node_vec1_prime, node_vec2_prime

class BigST(nn.Module):
    def __init__(self, num_nodes, T_out=12, in_dim=3,
                 hid_dim=32, num_layers=2, dropout=0.3,
                 tau=1.0, random_feature_dim=64,
                 use_residual=True, use_bn=True, use_long=False,
                 time_num=288, week_num=7):
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

        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, self.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, self.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        self.input_emb_layer = nn.Conv2d(T_out * in_dim, hid_dim, kernel_size=(1, 1), bias=True)

        self.W_1 = nn.Conv2d(self.node_dim + self.time_dim * 2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(self.node_dim + self.time_dim * 2, hid_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linear_conv.append(
                linearized_conv(hid_dim*4, hid_dim*4, dropout, tau, random_feature_dim)
            )
            self.bn.append(nn.LayerNorm(hid_dim*4))

        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim*4*2 + hid_dim + T_out, T_out, kernel_size=(1, 1), bias=True)
        else:
            self.regression_layer = nn.Conv2d(hid_dim*4*2, T_out, kernel_size=(1, 1), bias=True)

    def forward(self, x, feat=None):
        # x: (B,N,T_in,3)
        B, N, T, D = x.size()
        time_idx = (x[:, :, -1, 1] * self.time_num).long().clamp(0, self.time_num-1)
        week_idx = x[:, :, -1, 2].long().clamp(0, self.week_num-1)

        time_emb = self.time_emb_layer[time_idx.to(self.time_emb_layer.device)]
        week_emb = self.week_emb_layer[week_idx.to(self.week_emb_layer.device)]

        x_flat = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)
        input_emb = self.input_emb_layer(x_flat)

        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
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
                x = self.bn[i](x.permute(0,2,3,1)).permute(0,3,1,2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)
        x = self.activation(x)

        if self.use_long and feat is not None:
            feat = feat.permute(0,2,1).unsqueeze(-1)
            x = torch.cat([x, feat], dim=1)

        x = self.regression_layer(x).squeeze(-1)
        return x, 0.0
