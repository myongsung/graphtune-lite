import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Gemma3ForecastModel(nn.Module):
    """
    google/gemma-3-270m 을 backbone 으로 쓰는 시계열 예측 모델.

    입력:  x  [B, T_in, N]
    출력:  y  [B, T_out, N]

    추가 기능:
      - 그래프 인코딩 (adjacency 기반 이웃 평균)
      - coords 기반 노드 가중치
      - 시간 pooling (last / mean / max / attn)
      - sequence adapter (작은 bottleneck MLP)
      - backbone 전체 fine-tuning (freeze_backbone=False)
    """

    def __init__(
        self,
        num_nodes: int,
        T_in: int,
        T_out: int,
        hf_model_name: str = "google/gemma-3-270m",
        freeze_backbone: bool = False,      # 🔥 기본값: 전체 파인튜닝
        dropout: float = 0.1,
        # --- 그래프 관련 ---
        A=None,                             # adjacency matrix [N, N]
        use_graph_encoder: bool = True,
        coords=None,                        # [N, d]
        use_coords: bool = True,
        # --- 시간 pooling ---
        temporal_pooling: str = "attn",     # "last" | "mean" | "max" | "attn"
        # --- adapter ---
        adapter_dim: int = 64,              # 0 또는 None이면 비활성
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.T_in = T_in
        self.T_out = T_out
        self.temporal_pooling = temporal_pooling

        # 0) Gemma-3 backbone 로딩
        self.backbone = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        hidden_size = self.backbone.config.hidden_size

        # 1) 그래프 인코더 준비 (A normalize)
        self.use_graph_encoder = use_graph_encoder and (A is not None)
        if self.use_graph_encoder:
            A = torch.as_tensor(A, dtype=torch.float32)
            N = A.size(0)
            I = torch.eye(N, device=A.device)
            A_hat = A + I  # self loop 포함
            deg = A_hat.sum(dim=-1)  # [N]
            deg_inv_sqrt = (deg + 1e-6).pow(-0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # [N, N]
            self.register_buffer("A_norm", A_norm)
        else:
            self.A_norm = None

        # 2) coords 기반 노드 weight
        self.use_coords = use_coords and (coords is not None)
        if self.use_coords:
            coords_t = torch.as_tensor(coords, dtype=torch.float32)  # [N, d]
            self.register_buffer("coords_tensor", coords_t)
            coord_dim = coords_t.size(-1)
            self.coord_mlp = nn.Sequential(
                nn.Linear(coord_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),  # scalar weight per node
            )
        else:
            self.coords_tensor = None
            self.coord_mlp = None

        # 3) 입력 proj: [B, T_in, N] -> [B, T_in, H]
        self.input_proj = nn.Linear(num_nodes, hidden_size)

        # 4) sequence adapter (LoRA 비슷한 bottleneck MLP)
        if adapter_dim is not None and adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(hidden_size, adapter_dim),
                nn.ReLU(),
                nn.Linear(adapter_dim, hidden_size),
            )
        else:
            self.adapter = None

        # 5) 시간 pooling용 query (attn 모드일 때)
        if temporal_pooling == "attn":
            self.time_query = nn.Parameter(torch.randn(hidden_size))
        else:
            self.time_query = None

        # 6) 출력 proj: [B, H] -> [B, T_out * N]
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, T_out * num_nodes),
        )

        # 7) backbone freeze 옵션
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # freeze_backbone=False 이면 기본값(전체 학습) 그대로 둠

    # -------------------------------------------------------
    # forward
    # -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T_in, N]
        return : [B, T_out, N]
        """
        B, T, N = x.shape
        assert T == self.T_in and N == self.num_nodes

        # 0) 그래프 인코딩: self + neighbor 평균
        if self.use_graph_encoder and self.A_norm is not None:
            # x: [B, T, N], A_norm: [N, N]
            x_neigh = torch.einsum("btn,nm->btm", x, self.A_norm)  # 이웃 집계
            x = 0.5 * (x + x_neigh)

        # 1) coords 기반 노드 weight
        if self.use_coords and self.coords_tensor is not None:
            node_w = self.coord_mlp(self.coords_tensor)  # [N, 1]
            node_w = node_w.view(1, 1, N)               # [1, 1, N]
            x = x * node_w

        # 2) 입력 proj: [B, T_in, N] → [B, T_in, H] (float32)
        h = self.input_proj(x)

        # 3) Gemma backbone dtype 에 맞춤
        backbone_dtype = next(self.backbone.parameters()).dtype
        h_for_backbone = h.to(backbone_dtype)

        # 4) Gemma backbone 통과
        outputs = self.backbone.model(
            inputs_embeds=h_for_backbone,
            use_cache=False,
            output_hidden_states=False,
        )
        seq_hidden = outputs.last_hidden_state  # [B, T_in, H], half
        seq_hidden = seq_hidden.to(h.dtype)     # 다시 float32

        # 5) adapter residual
        if self.adapter is not None:
            seq_hidden = seq_hidden + self.adapter(seq_hidden)

        # 6) 시간 pooling
        if self.temporal_pooling == "last":
            summary = seq_hidden[:, -1, :]  # [B, H]
        elif self.temporal_pooling == "mean":
            summary = seq_hidden.mean(dim=1)
        elif self.temporal_pooling == "max":
            summary, _ = seq_hidden.max(dim=1)
        elif self.temporal_pooling == "attn" and self.time_query is not None:
            q = self.time_query.to(seq_hidden.dtype)  # [H]
            scores = torch.einsum("bth,h->bt", seq_hidden, q) / math.sqrt(seq_hidden.size(-1))
            attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
            summary = (seq_hidden * attn).sum(dim=1)           # [B, H]
        else:
            summary = seq_hidden[:, -1, :]

        # 7) 출력 proj: [B, H] → [B, T_out, N]
        y_hat = self.out_proj(summary)                # [B, T_out * N]
        y_hat = y_hat.view(B, self.T_out, self.num_nodes)

        # 8) NaN/Inf 제거 (half 연산 폭주 안전장치)
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)

        return y_hat
