import torch

from ..core._state_dict_utils import (
    _load_local_sgd_state_dict,
    _local_sgd_state_dict,
    _pre_state_dict_hook,
    _save_local_sgd_state_dict,
)
from ._init_utils import fsdp_inits
from ._runtime_utils import _init_streams, _reduce_grad, _share_state_and_init_handle_attrs, _unshard, forward


def patch_local_sgd_to_fsdp():
    torch.distributed.fsdp._runtime_utils._share_state_and_init_handle_attrs = _share_state_and_init_handle_attrs
    torch.distributed.fsdp._runtime_utils._init_streams = _init_streams
    torch.distributed.fsdp._runtime_utils._reduce_grad = _reduce_grad
    torch.distributed.fsdp._runtime_utils._unshard = _unshard

    torch.distributed.fsdp._state_dict_utils._pre_state_dict_hook = _pre_state_dict_hook

    torch.distributed.fsdp.FullyShardedDataParallel.__init__ = fsdp_inits
    torch.distributed.fsdp.FullyShardedDataParallel.forward = forward
    torch.distributed.fsdp.FullyShardedDataParallel.local_sgd_state_dict = staticmethod(_local_sgd_state_dict)
    torch.distributed.fsdp.FullyShardedDataParallel.save_local_sgd_state_dict = staticmethod(_save_local_sgd_state_dict)
    torch.distributed.fsdp.FullyShardedDataParallel.load_local_sgd_state_dict = staticmethod(_load_local_sgd_state_dict)
