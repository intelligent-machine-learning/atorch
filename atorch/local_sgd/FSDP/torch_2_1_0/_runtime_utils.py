# MODIFIED from torch.distributed.fsdp._runtime_utils
import logging
from typing import Any, Dict, Set, no_type_check

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _is_composable, _no_dispatch_record_stream
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp._runtime_utils import (
    HOMOGENEOUS_ATTR_NAMES,
    _accumulate_sharded_grad,
    _cast_buffers_to_dtype_and_device,
    _div_if_needed,
    _get_buffers_and_dtypes_for_computation,
    _get_reduce_scatter_tensors,
    _init_device_mesh,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _post_reduce_grad_callback,
    _pre_forward,
    _pre_forward_unshard,
    _reset_flat_param_grad_info_if_needed,
    _root_cast_forward_input,
    _validate_and_get_hybrid_shard_state,
    _wait_for_computation_stream,
)
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy, HandleTrainingState
from torch.distributed.utils import _p_assert, _to_kwargs

from atorch.local_sgd.FSDP.core._runtime_utils import _sync_if_needed, _sync_sharded_params

logger = logging.getLogger(__name__)

SYNC_MODE_CONTROL = False


@no_type_check
def _share_state_and_init_handle_attrs(
    root_state: _FSDPState,
    root_module: nn.Module,
) -> None:
    """
    Shares data structure state from the ``root_state`` to all FSDP states in
    ``root_module`` 's module tree, and initializes handle attributes. These
    are done together to require a single loop over the states.
    """
    handle = root_state._handle
    if handle:
        handle.init_flat_param_attributes()
    _validate_and_get_hybrid_shard_state(root_module)
    attr_name_to_values: Dict[str, Set[Any]] = {}
    for attr_name in HOMOGENEOUS_ATTR_NAMES:
        attr_name_to_values[attr_name] = set()
    root_state._all_handles = root_state._exec_order_data.all_handles  # share reference
    root_state._device_mesh = _init_device_mesh(root_state)
    # Update _has_optim_in_backward for each handle.
    for handle in root_state._all_handles:
        flat_param = handle.flat_param
        if hasattr(flat_param, "_in_backward_optimizers"):
            raise RuntimeError("FSDP optimizer in backward only supported with use_orig_params=True!")
        handle._has_optim_in_backward = flat_param._params is not None and any(
            hasattr(param, "_in_backward_optimizers") for param in flat_param._params
        )
    for fsdp_state in root_state._all_fsdp_states:
        for attr_name in HOMOGENEOUS_ATTR_NAMES:
            _p_assert(
                hasattr(fsdp_state, attr_name),
                f"FSDP state missing attribute {attr_name}",
            )
            attr_name_to_values[attr_name].add(getattr(fsdp_state, attr_name))
        if fsdp_state is root_state:
            continue
        # Relax the assert for non-root FSDP instances in case the nested
        # initialized module is wrapped again in FSDP later (e.g. after
        # training to run inference)
        _p_assert(
            fsdp_state._is_root is None or not fsdp_state._is_root,
            "Non-root FSDP instance's `_is_root` should not have been " "set yet or should have been set to `False`",
        )
        fsdp_state._is_root = False
        fsdp_state._unshard_stream = root_state._unshard_stream
        fsdp_state._post_backward_stream = root_state._post_backward_stream
        fsdp_state._pre_unshard_stream = root_state._pre_unshard_stream
        fsdp_state._all_reduce_stream = root_state._all_reduce_stream
        # HACK Share the average stream across FSDP instances.
        fsdp_state._average_stream = root_state._average_stream
        fsdp_state._default_stream = root_state._default_stream
        fsdp_state._exec_order_data = root_state._exec_order_data
        fsdp_state._free_event_queue = root_state._free_event_queue
        fsdp_state._device_mesh = root_state._device_mesh
        handle = fsdp_state._handle
        if handle:
            handle.init_flat_param_attributes()
    for attr_name, attr_values in attr_name_to_values.items():
        if len(attr_values) != 1:
            raise ValueError(f"Expects one homogeneous value for {attr_name} but got {attr_values}")


@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES for fsdp_state in state._all_fsdp_states
    )
    # HACK Determine whether to use local sgd
    uses_local_sgd = any(fsdp_state.use_local_sgd for fsdp_state in state._all_fsdp_states)
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # HACK Add and modify some streams.
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    # Stream to run HSDP's average parameters as async (if using HSDP)
    state._average_stream = (
        state._device_handle.Stream() if (uses_hybrid_sharding and uses_local_sgd) else state._default_stream
    )


@no_type_check
def _unshard(
    state: _FSDPState,
    handle: FlatParamHandle,
    unshard_stream: torch.Stream,
    pre_unshard_stream: torch.Stream,
) -> None:
    """
    Unshards the handles in ``handles``. If the handles are in
    :meth:`summon_full_params` and are using mixed precision, then they are
    forced to full precision.

    Postcondition: handle's ``FlatParameter`` 's data is the padded
    unsharded flat parameter on the compute device.
    """
    if not handle:
        return
    # HACK The main code for averaging parameters among nodes.
    if SYNC_MODE_CONTROL and handle._training_state == HandleTrainingState.FORWARD:
        _sync_sharded_params(state, handle)
        _wait_for_computation_stream(
            state._average_stream,
            unshard_stream,
            pre_unshard_stream,
        )
    with state._device_handle.stream(pre_unshard_stream):
        ran_pre_unshard = handle.pre_unshard()
    if ran_pre_unshard:
        unshard_stream.wait_stream(pre_unshard_stream)
    if state.limit_all_gathers:
        event = state._free_event_queue.dequeue_if_needed()
        if event:
            with torch.profiler.record_function("FullyShardedDataParallel.rate_limiter"):
                event.synchronize()
    with state._device_handle.stream(unshard_stream):
        handle.unshard()
        handle.post_unshard()


@no_type_check
def _reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """
    flat_param = handle.flat_param
    # HACK Fobidden grads all-reduce among nodes if `use_local_sgd` and accumulate `global_step`
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    if state.use_local_sgd:
        uses_hybrid_sharded_strategy = (
            uses_hybrid_sharded_strategy and not state.global_step >= state.local_sgd_warmup_steps
        )
        state.temp_step += 1
        if not state.temp_step % state.gradient_accumulation_steps:
            state.global_step += 1
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        dist.reduce_scatter_tensor(
            new_sharded_grad,
            padded_unsharded_grad,
            group=state.process_group,
        )
        if uses_hybrid_sharded_strategy:
            state._all_reduce_stream.wait_stream(state._post_backward_stream)
            with state._device_handle.stream(state._all_reduce_stream):
                # Since the new sharded gradient is produced in the post-
                # backward stream and consumed in the all-reduce stream,
                # inform the caching allocator
                _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
                _post_reduce_grad_callback(state, handle, grad_to_offload)
                return
        _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
        # NOTE: HSDP variants do not support communication hook.
    grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    _post_reduce_grad_callback(state, handle, grad_to_offload)


@no_type_check
def _root_pre_forward(
    state: _FSDPState,
    module: nn.Module,
    args,
    kwargs,
) -> None:
    """
    Runs pre-forward logic specific to the root FSDP instance, which should run
    before any individual module's pre-forward. This starts with an attempt at
    lazy initialization (which only runs non-vacuously once). Otherwise, if
    this is called on a non-root FSDP instance, then it returns directly.

    Args:
        module (nn.Module): Module for which this logic tries to run. It may or
            may not be the root. If not, then this method does not do anything.
    """
    with torch.profiler.record_function("FullyShardedDataParallel._root_pre_forward"):
        _lazy_init(state, module)
        _p_assert(state._is_root is not None, "Expects a root FSDP to have been set")
        if not state._is_root:
            # Always cast forward inputs in the root of this local FSDP unit for mixed
            # precision, as this is where mixed precision could be configed.
            # This is more useful for auto wrapping that is recommended in composable path.
            # For manual wrapping, cast forward inputs on each local FSDP unit root will
            # increase some overhead, so not turned on for model wrapper path right now where
            # manual wrapping is more broadly used.
            if _is_composable(state):
                return _root_cast_forward_input(state, module, args, kwargs)
            return args, kwargs

        # We cast buffers back to full precision if we're forcing full precision. Disjointly, we check if buffers
        # are in full precision and if we should cast them back to lower precision, which happens when
        # exiting eval() mode.
        handle = state._handle
        if handle:
            should_cast_buffers_to_full_prec = handle._force_full_precision
        else:
            should_cast_buffers_to_full_prec = True

        if should_cast_buffers_to_full_prec:
            _cast_buffers_to_dtype_and_device(
                buffers=dict(module.named_buffers()).values(),
                buffer_dtypes=list(state._buffer_name_to_orig_dtype.values()),
                device=state.compute_device,
            )
            # This flag is only set when we cast buffers to full precision, to avoid the
            # CPU overhead that can stem from retrieving all buffers and their types in the
            # following else branch.
            state._needs_buffer_dtype_restore_check = True
        elif getattr(state, "_needs_buffer_dtype_restore_check", False):
            # Check if buffers are in full precision and we need to cast them
            # back down.
            (
                buffers,
                buffer_dtypes_for_computation,
            ) = _get_buffers_and_dtypes_for_computation(state, module)
            if len(buffers) > 0 and len(buffer_dtypes_for_computation) > 0:
                if any(
                    buffer.dtype != buffer_dtype_for_computation
                    for buffer, buffer_dtype_for_computation in zip(buffers, buffer_dtypes_for_computation)
                ):
                    # Assume we have to cast everything if there is one mismatch
                    _cast_buffers_to_dtype_and_device(buffers, buffer_dtypes_for_computation, state.compute_device)
            # We don't have to check this again until we cast buffers to full precision again.
            state._needs_buffer_dtype_restore_check = False

        if state.forward_prefetch:
            handles = []
            for fsdp_state in state._all_fsdp_states:
                if fsdp_state._handle:
                    handles.append(fsdp_state._handle)
            for handle in handles:
                handle._needs_pre_forward_unshard = True
        # HACK Control the calculation streams corresponding to local sgd
        global SYNC_MODE_CONTROL
        SYNC_MODE_CONTROL = _sync_if_needed(state, module)
        if SYNC_MODE_CONTROL:
            state._average_stream.wait_stream(state._device_handle.current_stream())
        else:
            _wait_for_computation_stream(
                state._device_handle.current_stream(),
                state._unshard_stream,
                state._pre_unshard_stream,
            )
        _reset_flat_param_grad_info_if_needed(state._all_handles)

        # Prepares the forward inputs by moving them to ``compute_device``
        # TODO: Do not use the side stream for tensor copies for now; investigate
        # the perf with/without it.
        with torch.profiler.record_function("FullyShardedDataParallel._to_kwargs"):
            args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, state.compute_device, False)
        args = args_tuple[0]
        kwargs = kwargs_tuple[0]

        return _root_cast_forward_input(state, module, args, kwargs)


# HACK Use `_root_pre_forward` and `_pre_forward` modified here.
def forward(self, *args: Any, **kwargs: Any) -> Any:
    """
    Runs the forward pass for the wrapped module, inserting FSDP-specific
    pre- and post-forward sharding logic.
    """
    handle = self._handle
    with torch.autograd.profiler.record_function("FullyShardedDataParallel.forward"):
        args, kwargs = _root_pre_forward(self, self, args, kwargs)
        unused = None
        args, kwargs = _pre_forward(
            self,
            handle,
            _pre_forward_unshard,
            self._fsdp_wrapped_module,
            args,
            kwargs,
        )
        if handle:
            _p_assert(
                handle.flat_param.device == self.compute_device,
                "Expected `FlatParameter` to be on the compute device "
                f"{self.compute_device} but got {handle.flat_param.device}",
            )
        output = self._fsdp_wrapped_module(*args, **kwargs)
        return _post_forward(self, handle, _post_forward_reshard, self, unused, output)
