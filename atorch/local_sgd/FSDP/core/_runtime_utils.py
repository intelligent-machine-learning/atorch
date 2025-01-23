# MODIFIED from torch.distributed.fsdp._runtime_utils
import logging
import time
from typing import Optional, no_type_check

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState

logger = logging.getLogger(__name__)

try:
    from torch.distributed.fsdp.flat_param import FlatParamHandle
except Exception as e:
    logger.debug(f"Meet Exception: {e}!")
    from torch.distributed.fsdp._flat_param import FlatParamHandle


def _lazy_init_outer_optimizer(
    handle: Optional[FlatParamHandle],
    cpu_init: bool = False,
) -> None:
    if cpu_init:
        cpu_flat_param = handle.flat_param.cpu().detach().clone()
        pinned_cpu_flat_param = cpu_flat_param.pin_memory()  # for non-blocking copy
        handle.last_synced_params = torch.nn.Parameter(pinned_cpu_flat_param)
    else:
        handle.last_synced_params = torch.nn.Parameter(handle.flat_param.detach().clone())
    if handle.outer_optim_class is not None:
        handle.outer_optimizer = handle.outer_optim_class([handle.last_synced_params], **handle.outer_optim_kwargs)
    else:
        handle.outer_optimizer = None


def _sync_if_needed(
    state: _FSDPState,
    module: nn.Module,
) -> bool:
    if not module.training:
        return False
    if not state.use_local_sgd:
        return False
    if not state._handle.flat_param.requires_grad:
        return False
    if state.temp_step % state.gradient_accumulation_steps:  # only average once for the same `global_step`
        return False
    if state.global_step < state.local_sgd_warmup_steps:
        if state.use_async:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            state.last_total_global_step = total_global_step.item()
        return False
    if state.use_async:
        current_time = time.time() / dist.get_world_size(state.process_group)
        current_time = torch.tensor(current_time).to(state.compute_device)
        dist.all_reduce(current_time, group=state.process_group)
        current_time = current_time.item()
        if state.last_sync_timestamp is None:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            state.last_total_global_step = total_global_step.item()
            state.last_sync_timestamp = current_time
            return True
        elif current_time - state.last_sync_timestamp < state.local_sgd_sync_time:
            return False
        else:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            diff_global_steps = total_global_step.item() - state.last_total_global_step
            if diff_global_steps < state.min_total_global_steps:
                # Should we skip this synchronization or keep looping until the synchronization is completed?
                state.last_sync_timestamp = current_time
                return False
            else:
                state.last_sync_timestamp = current_time
                state.last_total_global_step = total_global_step.item()
                return True
    else:
        if (state.global_step - state.local_sgd_warmup_steps) % state.local_sgd_sync_interval:
            return False
        return True


@no_type_check
def _sync_sharded_params(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    with state._device_handle.stream(state._average_stream):
        pseudo_gradient = None
        if state.use_async and state.use_step_weight:
            total_global_step = torch.tensor(state.global_step).to(state.compute_device)
            dist.all_reduce(total_global_step, group=state._inter_node_pg)
            step_weight = state.global_step / total_global_step.item()

        if state._handle.gta_reducer is None and not state._handle.use_outer_optim:
            with torch.no_grad():
                if state.use_async and state.use_step_weight:
                    handle.flat_param.data *= step_weight
                else:
                    handle.flat_param.data /= dist.get_world_size(state._inter_node_pg)
                dist.all_reduce(handle.flat_param.data, group=state._inter_node_pg)
        else:
            if handle.last_synced_params is None:
                _lazy_init_outer_optimizer(handle)

        if state.local_sgd_cpu_offload and (handle.gta_reducer is not None or handle.use_outer_optim):
            param_device = handle.flat_param.device
            with torch.no_grad():
                handle.last_synced_params.data = handle.last_synced_params.data.to(param_device, non_blocking=True)
            if handle.use_outer_optim:
                for opt_state in handle.outer_optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k].data = v.data.to(param_device, non_blocking=True)

        if handle.gta_reducer is not None or handle.use_outer_optim:
            with torch.no_grad():
                pseudo_gradient = handle.last_synced_params.data - handle.flat_param.data
                if state.local_sgd_skip_anomaly or state.pseudo_gradnorm_reduce:
                    pseudo_gradnorm = torch.linalg.vector_norm(pseudo_gradient, 2).pow_(2)
                    dist.all_reduce(pseudo_gradnorm, group=state.process_group, op=torch.distributed.ReduceOp.SUM)
                    pseudo_gradnorm.pow_(0.5)

                if state.local_sgd_skip_anomaly:
                    gradnorm_value = pseudo_gradnorm
                    if handle.local_sgd_anomaly_detector.is_outlier(gradnorm_value):
                        # nullify this work's update
                        pseudo_gradient.mul_(0.0)
                        pseudo_gradnorm += 1e6
                        if state.is_debug and dist.get_rank(state.process_group) == 0:
                            logger.info(
                                f"Full Shard group [{dist.get_rank(state._inter_node_pg)}] step [{state.global_step}] ",
                                f"Pseudo Gradnorm: {gradnorm_value} deemed outlier",
                            )
                    handle.local_sgd_anomaly_detector.update(gradnorm_value)

                if handle.gta_reducer is None:
                    if state.use_async and state.use_step_weight:
                        pseudo_gradient *= step_weight
                    else:
                        pseudo_gradient /= dist.get_world_size(state._inter_node_pg)
                    dist.all_reduce(pseudo_gradient, group=state._inter_node_pg)
                else:
                    reducer_kwargs = {}
                    if state.pseudo_gradnorm_reduce:
                        # penalty on local copies with large grad norm
                        reducer_kwargs["weight"] = -pseudo_gradnorm
                    if state.use_async and state.use_step_weight:
                        # penalty on local copies with step weight
                        reducer_kwargs["step_weight"] = step_weight
                        reducer_kwargs["step_weight_ratio"] = state.step_weight_ratio
                    handle.gta_reducer.reduce_tensor(pseudo_gradient, **reducer_kwargs)
        if pseudo_gradient is not None:
            if state.clip_pseudo_grad is not None:
                total_norm = torch.linalg.vector_norm(pseudo_gradient, 2).pow_(2)
                dist.all_reduce(total_norm, group=state.process_group)
                total_norm.pow_(0.5)
                clip_coef = state.clip_pseudo_grad / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                pseudo_gradient.detach().mul_(clip_coef_clamped)
                if state.is_debug and dist.get_rank(state.process_group) == 0:
                    logger.info(
                        f"rank[{dist.get_rank()}]: "
                        f"total_norm:{total_norm}, "
                        f"clip_pseudo_grad:{state.clip_pseudo_grad}, "
                        f"clip_coef_clamped:{clip_coef_clamped}."
                    )
            if not (state.local_sgd_skip_anomaly and pseudo_gradient.abs().sum() < 1e-6):
                if handle.use_outer_optim:
                    handle.last_synced_params.grad = pseudo_gradient
                    handle.outer_optimizer.step()
                    handle.outer_optimizer.zero_grad()
                else:
                    handle.last_synced_params.data.sub_(pseudo_gradient)
            handle.flat_param.data.copy_(handle.last_synced_params.data)

        if state.local_sgd_cpu_offload:
            if handle.use_outer_optim:
                for opt_state in handle.outer_optimizer.state.values():
                    for k, v in opt_state.items():
                        if isinstance(v, torch.Tensor):
                            opt_state[k].data = v.data.to("cpu", non_blocking=True)
            if handle.gta_reducer is not None or handle.use_outer_optim:
                with torch.no_grad():
                    handle.last_synced_params.data = handle.last_synced_params.data.to("cpu", non_blocking=True)
