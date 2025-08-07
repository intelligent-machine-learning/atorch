import types
from typing import List

import torch

from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    print("********** megatron_lm_available: true **********")
    from megatron.core.dist_checkpointing.mapping import ShardedStateDict
    from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
    from megatron.core.transformer.module import param_is_not_shared
else:
    print("********** megatron_lm_available: false **********")
    from typing import Any, Dict

    ShardedStateDict = Dict[str, Any]

    def param_is_not_tensor_parallel_duplicate(param):
        """Returns true if the passed-in parameter is not a duplicate parameter
        on another TP rank."""
        return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel

    def param_is_not_shared(param):
        return not hasattr(param, "shared") or not param.shared


def zero_out_shard_fp32_memory(chained_optimizer):
    """
    Set the memory of all params in shard_fp32_groups and shard_fp32_from_float16_groups
    of each DistributedOptimizer in chained_optimizers to zero, but keep the objects.
    """
    print("********** zero_out_shard_fp32_memory **********")
    for opt in chained_optimizer.chained_optimizers:
        # DistributedOptimizer
        if not hasattr(opt, "shard_fp32_groups") or not hasattr(opt, "shard_fp32_from_float16_groups"):
            continue
        # shard_fp32_groups
        for group in opt.shard_fp32_groups:
            for param in group:
                if isinstance(param, torch.Tensor):
                    param.data = torch.empty((1,), dtype=param.dtype, device=param.device)
        # shard_fp32_from_float16_groups
        for group in opt.shard_fp32_from_float16_groups:
            for param in group:
                if isinstance(param, torch.Tensor):
                    param.data = torch.empty((1,), dtype=param.dtype, device=param.device)


def patch_distributed_optimizer(distributed_optimizer):
    def virtual_get_main_grads_for_grad_norm(self) -> List[torch.Tensor]:
        """
        Get main_grads that should be taken into account to compute the grad norm.
        Filter parameters based on:
          - grad should not be None.
          - parameter should not be shared (i.e., grads shouldn't be double counted while
            computing norms).
          - should not be a replica due to tensor model parallelism.
        """
        print("********** virtual_distributed_optimizer get_main_grads_for_grad_norm**********")

        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            if self.config.use_precision_aware_optimizer:
                grad = param.decoupled_grad if hasattr(param, "decoupled_grad") else None
            else:
                grad = param.virtual_grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)

            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

    def virtual_copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """
        print("********** virtual_distributed_optimizer copy model grads to main grads **********")
        if self.is_stub_optimizer:
            return

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    # assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    if self.config.use_precision_aware_optimizer:
                        # Pytorch requires a param and its' grad to be the same dtype, but we want
                        # their types to be different in precision-aware optimizer. So we use
                        # ".decoupled_grad" to replace ".grad".
                        # Note that this requires corresponding modifications in the optimizer (Let
                        # the optimizer read gradients from ".decoupled_grad" instead of ".grad").
                        shard_main_param.decoupled_grad = shard_model_grad
                    else:
                        # use grad will cause assgin error because grad is zeroed out in zero_out_shard_fp32_memory
                        shard_main_param.virtual_grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        if self.config.use_precision_aware_optimizer:
            copy_group_grads(self.model_float16_groups, self.shard_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)
        else:
            copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)

    distributed_optimizer.get_main_grads_for_grad_norm = types.MethodType(
        virtual_get_main_grads_for_grad_norm, distributed_optimizer
    )

    distributed_optimizer._copy_model_grads_to_main_grads = types.MethodType(
        virtual_copy_model_grads_to_main_grads, distributed_optimizer
    )


def patch_chained_optimizer(chained_optimizer):
    @torch.no_grad()
    def virtual_step(self):
        print("********** virtual_chained_optimizer step**********")

        return True, 0.0, 0

    def virtual_load_state_dict(self, state_dict):
        print("********** virtual_chained_optimizer load_state_dict**********")

        pass

    def virtual_sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        sharding_type: str = "fully_sharded_model_space",
    ):
        print("********** virtual_chained_optimizer sharded_state_dict**********")

        return {}

    def virtual_reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        print("********** virtual_chained_optimizer reload_model_params**********")

        pass

    def virtual_load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        print("********** virtual_chained_optimizer load_parameter_state**********")

        pass

    chained_optimizer.step = types.MethodType(virtual_step, chained_optimizer)
    chained_optimizer.load_state_dict = types.MethodType(virtual_load_state_dict, chained_optimizer)
    chained_optimizer.sharded_state_dict = types.MethodType(virtual_sharded_state_dict, chained_optimizer)
    chained_optimizer.reload_model_params = types.MethodType(virtual_reload_model_params, chained_optimizer)
    chained_optimizer.load_parameter_state = types.MethodType(virtual_load_parameter_state, chained_optimizer)


def virtual_distributed_optimizer_load_state_dict(self, state_dict):
    print("********** virtual_distributed_optimizer load_state_dict **********")

    pass
