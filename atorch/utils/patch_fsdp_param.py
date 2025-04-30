from typing import Any, Callable, Optional

import torch
from torch import nn

from atorch.common.constants import FSDPConstants
from atorch.common.env import EnvSetting
from atorch.common.singleton import SingletonMeta
from atorch.utils.version import torch_version

_FSDP2_PATCH_TORCH_VERSION = (2, 5, 0)

if torch_version() >= _FSDP2_PATCH_TORCH_VERSION:  # type: ignore
    try:
        from torch.distributed._composable.fsdp import _fsdp_init
        from torch.distributed._composable.fsdp._fsdp_param import FSDPParam
        from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup
        from torch.distributed._composable.fsdp._fsdp_state import FSDPState
    except (ImportError, ModuleNotFoundError):
        from torch.distributed.fsdp._fully_shard import _fsdp_init
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torch.distributed._tensor.api import DTensor
else:
    FSDPParam = object


class FSDP2PatchContext(metaclass=SingletonMeta):
    ORIGINAL_FSDP_STATE_PRE_BACKWARD: Optional[Callable] = None
    ORIGINAL_FSDP_PARAM_GROUP_BACKWARD_PREFETCH: Optional[Callable] = None
    ORIGINAL_FSDP_STATE_POST_FORWARD: Optional[Callable] = None
    ORIGINAL_FSDP_STATE_PRE_FORWARD: Optional[Callable] = None
    ORIGINAL_GET_MANAGED_STATES: Optional[Callable] = None
    ORIGINAL_INIT_SHARDED_PARAM: Optional[Callable] = None
    FSDP2_PATCH_TORCH_VERSION = _FSDP2_PATCH_TORCH_VERSION


def patch_fsdp2_init_sharded_param():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore

    if FSDP2PatchContext().ORIGINAL_INIT_SHARDED_PARAM is not None:
        return

    @torch.no_grad()
    def _atorch_init_sharded_param_wrapper(self, param: nn.Parameter, device: torch.device, shard_placement_fn=None):
        if torch_version() >= (2, 6, 0):  # type: ignore
            FSDP2PatchContext().ORIGINAL_INIT_SHARDED_PARAM(self, param, device, shard_placement_fn)
        else:
            FSDP2PatchContext().ORIGINAL_INIT_SHARDED_PARAM(self, param, device)

        if hasattr(param, FSDPConstants.CHECKPOINT_NAME):
            setattr(self.sharded_param, FSDPConstants.CHECKPOINT_NAME, param.checkpoint_name)
        if not hasattr(self.sharded_param, FSDPConstants.ATORCH_FSDP2_SHARDED):
            setattr(self.sharded_param, FSDPConstants.ATORCH_FSDP2_SHARDED, True)

    FSDP2PatchContext().ORIGINAL_INIT_SHARDED_PARAM = FSDPParam._init_sharded_param
    FSDPParam._init_sharded_param = _atorch_init_sharded_param_wrapper


def patch_fsdp2_get_managed_states():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore

    if FSDP2PatchContext().ORIGINAL_GET_MANAGED_STATES is not None:
        return

    def _atorch_get_managed_states_wrapper(modules):
        params, buffers = FSDP2PatchContext().ORIGINAL_GET_MANAGED_STATES(modules)
        selected_params = []
        for param in params:
            if not (isinstance(param, DTensor) and getattr(param, FSDPConstants.ATORCH_FSDP2_SHARDED, False)):
                selected_params.append(param)

        return selected_params, buffers

    FSDP2PatchContext().ORIGINAL_GET_MANAGED_STATES = _fsdp_init._get_managed_states
    _fsdp_init._get_managed_states = _atorch_get_managed_states_wrapper


def patch_fsdp2_pre_backward():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore

    if FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_BACKWARD is not None:
        return

    def _atorch_pre_backward_wrapper(self, grad: torch.Tensor) -> torch.Tensor:
        grad = FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_BACKWARD(self, grad)

        if hasattr(self, "_inter_state"):
            inter_ggm_state = getattr(self, "_inter_state")
            inter_ggm_state._pre_backward(None)
        return grad

    FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_BACKWARD = FSDPState._pre_backward
    FSDPState._pre_backward = _atorch_pre_backward_wrapper


def patch_fsdp2_backward_prefetch():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore

    if FSDP2PatchContext().ORIGINAL_FSDP_PARAM_GROUP_BACKWARD_PREFETCH is not None:
        return

    def _atorch_backward_prefetch_wrapper(self):
        if EnvSetting().CLOSE_FSDP2_BACKWARD_PREFETCH:
            return

        FSDP2PatchContext().ORIGINAL_FSDP_PARAM_GROUP_BACKWARD_PREFETCH(self)

    FSDP2PatchContext().ORIGINAL_FSDP_PARAM_GROUP_BACKWARD_PREFETCH = FSDPParamGroup._backward_prefetch
    FSDPParamGroup._backward_prefetch = _atorch_backward_prefetch_wrapper


def patch_fsdp2_post_forward():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore

    if FSDP2PatchContext().ORIGINAL_FSDP_STATE_POST_FORWARD is not None:
        return

    def _atorch_post_forward_wrapper(self, module: nn.Module, input: Any, output: Any):
        condition = (
            EnvSetting().CLOSE_FSDP2_BACKWARD_PREFETCH
            and torch.is_grad_enabled()
            and self._fsdp_param_group
            and (hasattr(self, "_inter_state") or getattr(self, "_is_inter_state", False))
        )
        if condition:
            old_post_forward_mesh_info = self._fsdp_param_group.post_forward_mesh_info
            self._fsdp_param_group.post_forward_mesh_info = None

        res = FSDP2PatchContext().ORIGINAL_FSDP_STATE_POST_FORWARD(self, module, input, output)

        if condition:
            self._fsdp_param_group.post_forward_mesh_info = old_post_forward_mesh_info

        return res

    FSDP2PatchContext().ORIGINAL_FSDP_STATE_POST_FORWARD = FSDPState._post_forward
    FSDPState._post_forward = _atorch_post_forward_wrapper


def patch_fsdp2_pre_forward():
    assert torch_version() >= FSDP2PatchContext().FSDP2_PATCH_TORCH_VERSION  # type: ignore
    if FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_FORWARD is not None:
        return

    def _atorch_pre_forward_wrapper(self, module, args, kwargs):
        condition = torch.is_grad_enabled() and getattr(self, "_is_inter_state", False)

        if condition:
            old_states_to_forward_prefetch = self._states_to_forward_prefetch
            self._states_to_forward_prefetch = []

        res = FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_FORWARD(self, module, args, kwargs)

        if condition:
            self._states_to_forward_prefetch = old_states_to_forward_prefetch

        return res

    FSDP2PatchContext().ORIGINAL_FSDP_STATE_PRE_FORWARD = FSDPState._pre_forward
    FSDPState._pre_forward = _atorch_pre_forward_wrapper
