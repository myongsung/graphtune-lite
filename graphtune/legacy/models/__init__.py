from .bigst import BigST
from .baselines import BaselineModel, HyperModel, DCRNNModel, DGCRNModel

def build_model(model_name, dataset_bundle, **kwargs):
    name = model_name.lower()
    A = dataset_bundle["A"]
    coords = dataset_bundle["coords"]
    N = dataset_bundle["num_nodes"]
    T_in = dataset_bundle["T_in"]
    T_out = dataset_bundle["T_out"]

    if name == "bigst":
        return BigST(num_nodes=N, T_out=T_out, **kwargs)
    if name == "baseline":
        return BaselineModel(A, T_in, T_out, **kwargs)
    if name == "hypernet":
        return HyperModel(A, coords, T_in, T_out, **kwargs)
    if name == "dcrnn":
        return DCRNNModel(A, N, T_in, T_out, **kwargs)
    if name == "dgcrn":
        return DGCRNModel(A, N, T_in, T_out, **kwargs)

    raise ValueError(f"Unknown model_name: {model_name}")
