from typing import Optional, no_type_check

from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
from torch.distributed.fsdp.api import ShardingStrategy

from atorch.local_sgd.configs import GTAConfig, LocalSGDConfig, OuterOptimizerConfig
from atorch.local_sgd.utils import GTAReducer, LinearReducer, OnlineDynamicEWMA


@no_type_check
def _init_local_sgd_state(
    state: _FSDPState,
    sharding_strategy: Optional[ShardingStrategy] = None,
    use_local_sgd: bool = False,
    local_sgd_config: Optional[LocalSGDConfig] = None,
    outer_optim_config: Optional[OuterOptimizerConfig] = None,
    gta_config: Optional[GTAConfig] = None,
) -> _FSDPState:
    if use_local_sgd and (sharding_strategy not in HYBRID_SHARDING_STRATEGIES):
        raise RuntimeError("Local SGD only supports hybrid sharding strategies.")
    if use_local_sgd:
        if local_sgd_config is None:
            raise ValueError("Must set local_sgd_config manually when using local sgd!")
        if local_sgd_config.local_sgd_warmup_steps < 0:
            raise ValueError(
                "Invalid local_sgd_warmup_steps value: {}.".format(local_sgd_config.local_sgd_warmup_steps)
            )
        if local_sgd_config.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps value: {}.".format(local_sgd_config.gradient_accumulation_steps)
            )
        if local_sgd_config.use_async:
            if local_sgd_config.local_sgd_sync_time < 0:
                raise ValueError("Invalid local_sgd_sync_time value: {}.".format(local_sgd_config.local_sgd_sync_time))
            if local_sgd_config.min_total_global_steps < 1:
                raise ValueError(
                    "Invalid min_total_global_steps value: {}.".format(local_sgd_config.min_total_global_steps)
                )
        else:
            if local_sgd_config.local_sgd_sync_interval < 1:
                raise ValueError(
                    "Invalid local_sgd_sync_interval value: {}.".format(local_sgd_config.local_sgd_sync_interval)
                )
    # Update _FSDPState's states
    ## Normal states
    state.use_local_sgd = use_local_sgd
    state.use_async = local_sgd_config.use_async
    state.is_debug = local_sgd_config.is_debug
    state.local_sgd_sync_interval = local_sgd_config.local_sgd_sync_interval
    state.local_sgd_warmup_steps = local_sgd_config.local_sgd_warmup_steps
    state.gradient_accumulation_steps = local_sgd_config.gradient_accumulation_steps
    state.clip_pseudo_grad = local_sgd_config.clip_pseudo_grad
    state.global_step = 0
    state.temp_step = 0
    state.local_sgd_cpu_offload = local_sgd_config.cpu_offload
    state.local_sgd_skip_anomaly = local_sgd_config.skip_anomaly
    state.weight_softmax_temperature = local_sgd_config.weight_softmax_temperature
    state.pseudo_gradnorm_reduce = local_sgd_config.pseudo_gradnorm_reduce
    if state.pseudo_gradnorm_reduce and state.weight_softmax_temperature is None:
        state.weight_softmax_temperature = 1.0
    ## Async related states
    state.local_sgd_sync_time = local_sgd_config.local_sgd_sync_time
    state.min_total_global_steps = local_sgd_config.min_total_global_steps
    state.use_step_weight = local_sgd_config.use_step_weight
    state.step_weight_ratio = local_sgd_config.step_weight_ratio
    state.last_sync_timestamp = None
    state.last_total_global_step = 0

    # Update handle's states
    ## Outer Optimizer states
    state._handle.outer_optim_class = outer_optim_config.outer_optim_class if outer_optim_config is not None else None
    state._handle.last_synced_params = None
    if use_local_sgd and state._handle.outer_optim_class is not None:
        state._handle.use_outer_optim = True
        state._handle.outer_optim_kwargs = outer_optim_config.outer_optim_kwargs
        state._handle.outer_optimizer = None
    else:
        state._handle.use_outer_optim = False
    ## GTA states
    state._handle.gta_reducer = None
    if gta_config is not None:
        if gta_config.reducer == "linear":
            state._handle.gta_reducer = LinearReducer(
                process_group=state._inter_node_pg,
                normalize=gta_config.normalize,
                weight_softmax_temperature=state.weight_softmax_temperature,
            )
        elif gta_config.reducer == "gta":
            state._handle.gta_reducer = GTAReducer(
                process_group=state._inter_node_pg,
                consensus_method=gta_config.consensus_method,
                sparsification_method=gta_config.sparsification_method,
                normalize=gta_config.normalize,
                density=gta_config.density,
                int8_mask=gta_config.int8_mask,
                weight_softmax_temperature=state.weight_softmax_temperature,
            )
    ## Anomaly Detector states
    if state.local_sgd_skip_anomaly:
        state._handle.local_sgd_anomaly_detector = OnlineDynamicEWMA(
            alpha=local_sgd_config.ewma_alpha,
            warmup_steps=local_sgd_config.ewma_warmup_steps,
            base_threshold=local_sgd_config.ewma_threshold,
        )

    return state
