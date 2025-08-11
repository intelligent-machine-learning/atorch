import inspect
import os

import torch

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.trainer_callback import AtorchTrainerCallback, AtorchTrainerControl, AtorchTrainerState
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core.package_info import __version__ as megatron_version
    from megatron.training import get_args

    from atorch.trainer.megatron import AtorchMegatronEngine


class DebugCallback(AtorchTrainerCallback):
    def __init__(self, train_engine):
        """
        Should be called after initializing Megatron.
        """
        super().__init__()
        self.train_engine: AtorchMegatronEngine = train_engine
        self.module_list = self.train_engine.module
        self.modules = self.train_engine.module[0].modules()
        self.args = get_args()

        self.activation_output_dir = os.path.join(self.args.save, f"activation_{megatron_version}")
        if self.args.rank == 0:
            os.makedirs(self.activation_output_dir, exist_ok=True)

        self.module_to_name = {}
        if self.args.rank == 0:
            for name, sub_module in self.train_engine.module[0].named_modules():
                m = inspect.getmodule(sub_module.__class__)
                if m is not None:
                    module_name = f"{m.__name__}.{sub_module._get_name()}"
                else:
                    module_name = f"{sub_module._get_name()}"
                logger.info(f"[Rank {self.args.rank}] module: {name} module_name: {module_name}")
                self.module_to_name[sub_module] = name

    def register_forward_pre_hook(self):
        def _hook(module: torch.nn.Module, args, kwargs):
            module_name = module._get_name()
            if self.args.rank == 0 and self.args.global_step == 0:
                logger.info(
                    f"[Rank {self.args.rank}] fwd pre  hook: {module_name} len(args) {len(args)} kwargs.keys() {kwargs.keys()}"  # noqa: E501
                )
                if module_name == "BailingMoeModel":
                    info = ""
                    for i, t in enumerate(args):
                        info += f" arg{i}: {t.shape if isinstance(t, torch.Tensor) else type(t)}"
                    logger.info(f"[Rank {self.args.rank}] fwd pre  hook: {module_name} {info}")  # noqa: E501

        return _hook

    def register_forward_post_hook(self):
        def _save(obj, name):
            path = os.path.join(
                self.activation_output_dir,
                f"step{self.args.global_step}_micro{self.args.micro_step}_rank{self.args.rank}_{name}.pth",
            )
            logger.info(f"[Rank {self.args.rank}] Saving activation to {path}")
            torch.save(obj, path)

        def _hook(module: torch.nn.Module, args, kwargs, result):
            module_name = module._get_name()
            if self.args.rank == 0 and self.args.global_step == 0:
                result_to_print = (
                    result.keys()
                    if isinstance(result, dict)
                    else result.shape
                    if isinstance(result, torch.Tensor)
                    else type(result)
                )
                logger.info(
                    f"[Rank {self.args.rank}] fwd post hook: {module_name} len(args) {len(args)} kwargs.keys() {kwargs.keys()} result: {result_to_print}"  # noqa: E501
                )
                if module_name == "BailingMoeModel":
                    info = ""
                    for i, t in enumerate(args):
                        info += f" arg{i}: {t.shape if isinstance(t, torch.Tensor) else type(t)}"
                    logger.info(f"[Rank {self.args.rank}] fwd post hook: {module_name} {info}")  # noqa: E501
            if self.args.rank == 0 and self.args.global_step in [2, 3, 4]:
                name: str = self.module_to_name[module]
                if "layers" in name:
                    splits = name.split(".")
                    layer_id = int(splits[4]) if len(splits) >= 5 else -1  # noqa: F841
                    if (
                        module_name
                        in [
                            "SelfAttention",
                            "MoELayer",
                            "TopKRouter",
                            "TEGroupedMLP",
                            "SharedExpertMLP",
                            "TransformerLayer",
                        ]
                        or "mlp" in name
                        or (
                            "self_attention" in name
                            and module_name not in ["UnfusedDotProductAttention", "FusedScaleMaskSoftmax", "Dropout"]
                        )
                    ):
                        _save(result, name)
                elif (
                    module_name not in ["DistributedDataParallel", "Float16Module"]
                    and "output_layer" not in module_name
                ):
                    _save(result, name)
                    if "final_layernorm" in name:
                        _save(args, name + "_input_args")
                        _save(kwargs, name + "_input_kwargs")

        return _hook

    def on_train_begin(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training, after Megatron initialization and building dataloader.
        """
        for m in self.modules:
            if isinstance(m, torch.nn.Module):
                # m.register_forward_pre_hook(self.register_forward_pre_hook(), with_kwargs=True)
                m.register_forward_hook(self.register_forward_post_hook(), with_kwargs=True)
