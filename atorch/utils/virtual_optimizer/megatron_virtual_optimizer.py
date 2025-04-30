from typing import Any, Callable, List, Optional

import torch

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.optimizer import (
        ChainedOptimizer,
        MegatronOptimizer,
        _get_param_groups,
        _update_min_and_max_lr_in_param_groups,
    )
    from megatron.core.optimizer.distrib_optimizer import MixedPrecisionOptimizer
    from megatron.core.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler, MegatronGradScaler
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    try:
        from transformer_engine.pytorch.optimizers import FusedAdam as Adam
        from transformer_engine.pytorch.optimizers import FusedSGD as SGD
    except ImportError:
        try:
            from apex.optimizers import FusedAdam as Adam
            from apex.optimizers import FusedSGD as SGD
        except ImportError:
            import warnings

            warnings.warn("Transformer Engine and Apex are not installed. Falling back to Torch optimizers.")
            from torch.optim import AdamW as Adam, SGD

    from megatron.core.transformer.module import MegatronModule


def _get_megatron_virtual_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    param_groups: List,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.

    Returns:
        Instance of MegatronOptimizer.
    """
    if config.optimizer == "adam":
        optimizer = Adam(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )

        def init_state_fn(opt):
            for group in opt.param_groups:
                for p in group["params"]:
                    if len(opt.state[p]) == 0:
                        opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                        opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

    elif config.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
        init_state_fn = None
    else:
        raise Exception("{} optimizer is not supported.".format(config.optimizer))

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        ]
        optimizer = MegatronVirtualOptimizer(*optimizer_args)
    else:
        raise ValueError("FP32 Virtual optimizer is not supported.")

    return optimizer


# Configure use_virtual_optimizer: true in antllm yaml file to enable this feature
def get_megatron_virtual_optimizer(
    train_args: AtorchTrainingArgs,
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(f"Setting up optimizer with {config}")

    # Collect param groups.
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_decoupled_learning_rate=config.decoupled_lr is not None,
    )

    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    dense_param_groups = list(filter(lambda g: not g["is_expert_parallel"], param_groups))
    moe_param_groups = list(filter(lambda g: g["is_expert_parallel"], param_groups))

    optimizers = [
        _get_megatron_virtual_optimizer_based_on_param_groups(
            config,
            param_groups=dense_param_groups,
        )
    ]
    if len(moe_param_groups) > 0:
        optimizers.append(
            _get_megatron_virtual_optimizer_based_on_param_groups(
                config,
                param_groups=moe_param_groups,
            )
        )

    if len(optimizers) == 1:
        return optimizers[0]

    return ChainedOptimizer(optimizers)


class MegatronVirtualOptimizer(MixedPrecisionOptimizer):
    def __init__(
        self,
        #  train_args: AtorchTrainingArgs,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable] = None,
    ):
        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        # self.train_args = train_args
        if hasattr(self.optimizer, "param_groups_master"):
            self.param_groups_master: List[Any] = []

    @torch.no_grad()
    def step(self):

        grad_norm = None
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)

        num_zeros_in_grad = None
        # self.optimizer.step()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        sharding_type: str = "fully_sharded_model_space",
    ):
        return {}

    def state_dict(self):
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        pass

    def zero_grad(self, set_to_none=True):
        pass
        # for groups in ():
        #     for group in groups:
        #         _zero_grad_group_helper(group, set_to_none)
