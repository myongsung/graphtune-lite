# graphtune/config.py
"""
v2 config.
Keep the same public constants as legacy for full compatibility.
"""

DEFAULT_MODEL_KWARGS = {
    "bigst": dict(hid_dim=32, num_layers=2, dropout=0.3),
    "baseline": dict(hidden_dim=32),
    "hypernet": dict(hidden_dim=32, hyper_hidden=64),
    "dcrnn": dict(hidden_dim=64, num_layers=2, K=2),
    "dgcrn": dict(hidden_dim=64, num_layers=2, K=2, emb_dim=10),

    "gemma3": dict(
        hf_model_name="google/gemma-3-270m",
        freeze_backbone=True,          # 처음엔 그대로 freeze 상태
        train_backbone_last_n=None,    # 예: 2 로 바꾸면 마지막 2개 레이어만 학습
        dropout=0.1,
        use_graph_encoder=True,
        use_coords=True,
        temporal_pooling="attn",       # "last", "mean", "max", "attn" 중 택
        adapter_dim=64,
    ),
    
}

__all__ = ["DEFAULT_MODEL_KWARGS"]
