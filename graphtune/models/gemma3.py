import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class Gemma3ForecastModel(nn.Module):
    """
    google/gemma-3-270m 을 backbone 으로 쓰는 시계열 예측 모델.

    입력:  x  [B, T_in, N]
    출력:  y  [B, T_out, N]
    """

    def __init__(
        self,
        num_nodes: int,
        T_in: int,
        T_out: int,
        hf_model_name: str = "google/gemma-3-270m",
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.T_in = T_in
        self.T_out = T_out

        # HF Gemma-3 로딩
        self.backbone = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        hidden_size = self.backbone.config.hidden_size

        # 입력: [B, T_in, N] → [B, T_in, H]
        self.input_proj = nn.Linear(num_nodes, hidden_size)

        # 출력: [B, H] → [B, T_out * N]
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, T_out * num_nodes),
        )

        # backbone freeze 옵션
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T_in, N]
        return : [B, T_out, N]
        """
        B, T, N = x.shape
        assert T == self.T_in and N == self.num_nodes

        # 1) 입력 투영: [B, T_in, N] → [B, T_in, H] (float32)
        h = self.input_proj(x)

        # 2) Gemma backbone dtype (보통 float16) 에 맞춤
        backbone_dtype = next(self.backbone.parameters()).dtype
        h_for_backbone = h.to(backbone_dtype)

        # 3) Gemma backbone 통과
        outputs = self.backbone.model(
            inputs_embeds=h_for_backbone,
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = outputs.last_hidden_state  # [B, T_in, H], half

        # 4) 마지막 토큰 summary → 다시 float32
        summary = last_hidden[:, -1, :].to(h.dtype)  # [B, H], float32

        # 5) 출력 proj: [B, H] → [B, T_out * N]
        y_hat = self.out_proj(summary)  # [B, T_out * N], float32
        y_hat = y_hat.view(B, self.T_out, self.num_nodes)

   
        # 7) NaN/Inf 제거
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)

        return y_hat
