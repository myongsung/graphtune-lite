import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_random_walk_matrices(A_np, eps=1e-6):
    A = torch.tensor(A_np, dtype=torch.float32)
    rowsum_fwd = A.sum(dim=1)
    inv_fwd = 1.0 / (rowsum_fwd + eps)
    W_fwd = torch.diag(inv_fwd) @ A

    AT = A.t()
    rowsum_bwd = AT.sum(dim=1)
    inv_bwd = 1.0 / (rowsum_bwd + eps)
    W_bwd = torch.diag(inv_bwd) @ AT
    return W_fwd, W_bwd

class BaselineModel(nn.Module):
    def __init__(self, A, T_in, T_out, hidden_dim=32):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32)
        N = A.shape[0]

        deg = A.sum(1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        A_norm = torch.diag(deg_inv_sqrt) @ A @ torch.diag(deg_inv_sqrt)
        self.register_buffer("A_norm", A_norm)

        self.num_nodes = N
        self.T_in = T_in
        self.T_out = T_out
        self.hidden_dim = hidden_dim

        self.temporal_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.node_out_weight = nn.Parameter(torch.randn(N, hidden_dim, T_out) * 0.01)
        self.node_out_bias = nn.Parameter(torch.zeros(N, T_out))

    def forward(self, x):
        B, T, N = x.shape
        x_perm = x.permute(0,2,1).contiguous().view(B*N,1,T)
        h = torch.relu(self.temporal_conv(x_perm))
        h = h.view(B, N, self.hidden_dim, T)[:, :, :, -1]

        h_agg = torch.relu(torch.einsum("ij,bjh->bih", self.A_norm, h))
        y = torch.einsum("bnh,nho->bno", h_agg, self.node_out_weight) + self.node_out_bias
        return y.permute(0,2,1).contiguous()

class HyperModel(nn.Module):
    def __init__(self, A, coords, T_in, T_out, hidden_dim=32, hyper_hidden=64):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32)
        N = A.shape[0]

        deg = A.sum(1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        A_norm = torch.diag(deg_inv_sqrt) @ A @ torch.diag(deg_inv_sqrt)
        self.register_buffer("A_norm", A_norm)

        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32))
        coord_dim = coords.shape[1]

        self.num_nodes = N
        self.T_in = T_in
        self.T_out = T_out
        self.hidden_dim = hidden_dim

        self.temporal_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.hyper = nn.Sequential(
            nn.Linear(coord_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, hidden_dim*T_out + T_out),
        )

    def forward(self, x):
        B, T, N = x.shape
        x_perm = x.permute(0,2,1).contiguous().view(B*N,1,T)
        h = torch.relu(self.temporal_conv(x_perm))
        h = h.view(B, N, self.hidden_dim, T)[:, :, :, -1]

        h_agg = torch.relu(torch.einsum("ij,bjh->bih", self.A_norm, h))
        Wb = self.hyper(self.coords)
        H, T_out = self.hidden_dim, self.T_out
        W = Wb[:, :H*T_out].view(N, H, T_out)
        b = Wb[:, H*T_out:].view(N, T_out)

        y = torch.einsum("bnh,nho->bno", h_agg, W) + b
        return y.permute(0,2,1).contiguous()

class DiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, K, W_fwd, W_bwd):
        super().__init__()
        self.K = K
        self.register_buffer("W_fwd", W_fwd)
        self.register_buffer("W_bwd", W_bwd)
        self.mlp = nn.Linear((2*K+1)*in_dim, out_dim)

    def forward(self, x):
        outs = [x]
        x_f = x
        for _ in range(self.K):
            x_f = torch.einsum("ij,bjf->bif", self.W_fwd, x_f); outs.append(x_f)
        x_b = x
        for _ in range(self.K):
            x_b = torch.einsum("ij,bjf->bif", self.W_bwd, x_b); outs.append(x_b)
        return self.mlp(torch.cat(outs, dim=-1))

class DCGRUCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, K, W_fwd, W_bwd):
        super().__init__()
        in_total = input_dim + hidden_dim
        self.diff_r = DiffusionConv(in_total, hidden_dim, K, W_fwd, W_bwd)
        self.diff_z = DiffusionConv(in_total, hidden_dim, K, W_fwd, W_bwd)
        self.diff_n = DiffusionConv(in_total, hidden_dim, K, W_fwd, W_bwd)

    def forward(self, x, h_prev):
        x_h = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.diff_r(x_h))
        z = torch.sigmoid(self.diff_z(x_h))
        x_rh = torch.cat([x, r*h_prev], dim=-1)
        n_tilde = torch.tanh(self.diff_n(x_rh))
        return (1.0 - z)*h_prev + z*n_tilde

class DCRNNModel(nn.Module):
    def __init__(self, A, num_nodes, T_in, T_out,
                 input_dim=1, hidden_dim=64, num_layers=2, K=2):
        super().__init__()
        self.num_nodes, self.T_in, self.T_out = num_nodes, T_in, T_out
        self.hidden_dim, self.num_layers, self.K = hidden_dim, num_layers, K

        W_fwd, W_bwd = compute_random_walk_matrices(A)
        self.register_buffer("W_fwd", W_fwd)
        self.register_buffer("W_bwd", W_bwd)

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_dim_l = input_dim if l == 0 else hidden_dim
            self.cells.append(DCGRUCell(num_nodes, in_dim_l, hidden_dim, K, W_fwd, W_bwd))

        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, y=None, teacher_forcing_ratio=0.0):
        B, T_in, N = x.shape
        x = x.unsqueeze(-1)
        h = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        for t in range(T_in):
            x_l = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l] = cell(x_l, h[l]); x_l = h[l]

        dec_input = x[:, -1]
        outs = []
        use_tf = self.training and (y is not None) and teacher_forcing_ratio > 0

        for t in range(self.T_out):
            x_l = dec_input
            for l, cell in enumerate(self.cells):
                h[l] = cell(x_l, h[l]); x_l = h[l]
            y_t = self.proj(x_l)
            outs.append(y_t)

            if use_tf and t < self.T_out-1 and torch.rand(1, device=x.device).item() < teacher_forcing_ratio:
                dec_input = y[:, t].unsqueeze(-1)
            else:
                dec_input = y_t

        return torch.stack(outs, dim=1).squeeze(-1)

class DynDiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.mlp = nn.Linear((2*K+1)*in_dim, out_dim)

    def forward(self, x, W_fwd, W_bwd):
        outs = [x]
        x_f = x
        for _ in range(self.K):
            x_f = torch.einsum("ij,bjf->bif", W_fwd, x_f); outs.append(x_f)
        x_b = x
        for _ in range(self.K):
            x_b = torch.einsum("ij,bjf->bif", W_bwd, x_b); outs.append(x_b)
        return self.mlp(torch.cat(outs, dim=-1))

class DGCRUCell(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, K):
        super().__init__()
        in_total = input_dim + hidden_dim
        self.diff_r = DynDiffusionConv(in_total, hidden_dim, K)
        self.diff_z = DynDiffusionConv(in_total, hidden_dim, K)
        self.diff_n = DynDiffusionConv(in_total, hidden_dim, K)

    def forward(self, x, h_prev, W_fwd, W_bwd):
        x_h = torch.cat([x, h_prev], dim=-1)
        r = torch.sigmoid(self.diff_r(x_h, W_fwd, W_bwd))
        z = torch.sigmoid(self.diff_z(x_h, W_fwd, W_bwd))
        x_rh = torch.cat([x, r*h_prev], dim=-1)
        n_tilde = torch.tanh(self.diff_n(x_rh, W_fwd, W_bwd))
        return (1.0 - z)*h_prev + z*n_tilde

class DGCRNModel(nn.Module):
    def __init__(self, A, num_nodes, T_in, T_out,
                 input_dim=1, hidden_dim=64, num_layers=2, K=2, emb_dim=10):
        super().__init__()
        self.num_nodes, self.T_in, self.T_out = num_nodes, T_in, T_out
        self.hidden_dim, self.num_layers, self.K = hidden_dim, num_layers, K

        W_fwd_fix, W_bwd_fix = compute_random_walk_matrices(A)
        self.register_buffer("W_fwd_fix", W_fwd_fix)
        self.register_buffer("W_bwd_fix", W_bwd_fix)

        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.node_emb2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_dim_l = input_dim if l == 0 else hidden_dim
            self.cells.append(DGCRUCell(num_nodes, in_dim_l, hidden_dim, K))

        self.proj = nn.Linear(hidden_dim, 1)

    def _compute_dynamic_rw(self):
        A_dyn = F.relu(self.node_emb1 @ self.node_emb2.t())
        W_fwd_dyn = A_dyn / (A_dyn.sum(1, keepdim=True) + 1e-6)
        A_dyn_T = A_dyn.t()
        W_bwd_dyn = A_dyn_T / (A_dyn_T.sum(1, keepdim=True) + 1e-6)
        a = torch.sigmoid(self.alpha)
        return a*self.W_fwd_fix + (1-a)*W_fwd_dyn, a*self.W_bwd_fix + (1-a)*W_bwd_dyn

    def forward(self, x):
        B, T_in, N = x.shape
        x = x.unsqueeze(-1)
        W_fwd, W_bwd = self._compute_dynamic_rw()

        h = [torch.zeros(B, N, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        for t in range(T_in):
            x_l = x[:, t]
            for l, cell in enumerate(self.cells):
                h[l] = cell(x_l, h[l], W_fwd, W_bwd); x_l = h[l]

        dec_input = x[:, -1]
        outs = []
        for _ in range(self.T_out):
            x_l = dec_input
            for l, cell in enumerate(self.cells):
                h[l] = cell(x_l, h[l], W_fwd, W_bwd); x_l = h[l]
            y_t = self.proj(x_l)
            outs.append(y_t)
            dec_input = y_t
        return torch.stack(outs, dim=1).squeeze(-1)
