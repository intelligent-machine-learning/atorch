import warnings

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
    from apex.parallel import SyncBatchNorm
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Try using apex LayerNorm but import fail:%s" % e)
    from torch.nn import LayerNorm as LayerNorm
    from torch.nn import SyncBatchNorm
from .layernorm import AtorchLayerNorm

__all__ = ["LayerNorm", "SyncBatchNorm", "AtorchLayerNorm"]
