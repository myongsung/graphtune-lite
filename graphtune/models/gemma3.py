# graphtune/models/gemma3.py
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM


class Gemma3ForecastModel(nn.Module):
    """
    google/gemma-3-270m 을 backbone 으로 쓰는 시계열 예측 모델.

    입력:  x  [B, T_in, N]
    출력:  y  [B, T_out, N]

    아이디어:
      - 각 time step 의 N개 노드 값을 Linear 로 H차원으로 투영 → [B, T_in, H]
      - Gemma-3의 Transformer backbone 에 inputs_embeds 로 입력
      - 마지막 hidden state (마지막 토큰) 을 summary 로 사용 → [B, H]
      - Linear 로 [B, T_out * N] 으로 매핑 후 reshape → [B, T_out, N]
    """

    def __init__(
        self,
        num_nodes: int,
        T_in: int,
        T_out: int,
        hf_model_name: str = "google/gemma-3-270m",  # ⬅ 이름 변경
        freeze_backbone: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.T_in = T_in
        self.T_out = T_out

        # 1) Hugging Face Gemma-3 로딩
        #    - CausalLM 이지만 우리는 backbone 의 transformer 부분만 사용.
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        # 대부분의 모델과 동일하게 hidden_size 에 차원이 들어있음
        hidden_size = self.backbone.config.hidden_size

        # 2) 입력: [B, T_in, N] → [B, T_in, H]
        self.input_proj = nn.Linear(num_nodes, hidden_size)

        # 3) 출력: [B, H] → [B, T_out * N]
        self.out_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, T_out * num_nodes),
        )

        # 4) backbone freeze 옵션 (원하면 LoRA/partial unfreeze 로 바꿀 수 있음)
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

        # 1) 노드 차원 → Gemma hidden 차원
        #    [B, T_in, N] → [B, T_in, H]
        h = self.input_proj(x)

        # 2) Gemma backbone 통과
        # 대부분의 HF CausalLM 모델에서 .model 은 내부 Transformer 입니다.
        # inputs_embeds 로 직접 시퀀스를 넣을 수 있음.
        outputs = self.backbone.model(
            inputs_embeds=h,
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = outputs.last_hidden_state  # [B, T_in, H]

        # 3) 마지막 토큰의 hidden 을 summary 로 사용
        summary = last_hidden[:, -1, :]  # [B, H]

        # 4) 예측값으로 투영
        y_hat = self.out_proj(summary)  # [B, T_out * N]
        y_hat = y_hat.view(B, self.T_out, self.num_nodes)

        return y_hat
