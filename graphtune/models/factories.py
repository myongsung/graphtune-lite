# graphtune/models/factories.py
from .registry import register_model
from .baselines import BaselineModel, HyperModel, DCRNNModel, DGCRNModel
from .bigst import BigST

@register_model("bigst")
def _build_bigst(bundle, **kwargs):
    return BigST(
        num_nodes=bundle["num_nodes"],
        T_in=bundle["T_in"],
        T_out=bundle["T_out"],
        **kwargs
    )

@register_model("baseline")
def _build_baseline(bundle, **kwargs):
    return BaselineModel(
        bundle["A"],
        bundle["T_in"],
        bundle["T_out"],
        **kwargs
    )

@register_model("hypernet")
def _build_hypernet(bundle, **kwargs):
    return HyperModel(
        bundle["A"],
        bundle["coords"],
        bundle["T_in"],
        bundle["T_out"],
        **kwargs
    )

@register_model("dcrnn")
def _build_dcrnn(bundle, **kwargs):
    return DCRNNModel(
        bundle["A"],
        bundle["num_nodes"],
        bundle["T_in"],
        bundle["T_out"],
        **kwargs
    )

@register_model("dgcrn")
def _build_dgcrn(bundle, **kwargs):
    return DGCRNModel(
        bundle["A"],
        bundle["num_nodes"],
        bundle["T_in"],
        bundle["T_out"],
        **kwargs
    )
