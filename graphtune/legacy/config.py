DEFAULT_MODEL_KWARGS = {
    "bigst": dict(hid_dim=32, num_layers=2, dropout=0.3),
    "baseline": dict(hidden_dim=32),
    "hypernet": dict(hidden_dim=32, hyper_hidden=64),
    "dcrnn": dict(hidden_dim=64, num_layers=2, K=2),
    "dgcrn": dict(hidden_dim=64, num_layers=2, K=2, emb_dim=10),
}
