import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from megatron.training import get_args
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import get_checkpoint_name
from transformers import TrainerControl, TrainerState, TrainingArguments

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.ckptsaver import CkptSaver
from atorch.trainer.utils import is_main_process
from atorch.utils.path_utils import path_is_empty


def move_not_empty_iteration_path_as_backup(iteration_path: Path):
    if iteration_path.exists() and not path_is_empty(iteration_path):
        from datetime import datetime

        try:
            last_folder = iteration_path.parts[-1]
            destination_dir_name = f"{last_folder}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            auto_trash = iteration_path.parent.joinpath("auto_trash")
            os.makedirs(str(auto_trash), exist_ok=True)
            destination_dir_path = auto_trash.joinpath(destination_dir_name)
            logger.info(f"trying to move not empty iteration dir {str(iteration_path)} to {str(destination_dir_path)}")
            shutil.move(str(iteration_path), str(destination_dir_path))
            # incase some save system will be slow when moving large dir. Use copy and delete instead.
            # shutil.copytree(str(iteration_path), str(destination_dir_path))
            # shutil.rmtree(str(iteration_path))
            logger.info(
                f"not empty iteration dir {str(iteration_path)} successfully moved to {str(destination_dir_path)}"
            )
        except Exception:
            logger.error(f"failed to move iteration dir {str(iteration_path)}, please check manually.")
            import traceback

            traceback.print_exc()


class MegatronCkptSaver(CkptSaver, ABC):
    def get_interation_path(self, output_dir: str, iteration: int, **kwargs) -> str:
        kwargs["return_base_dir"] = kwargs.get("return_base_dir", True)
        checkpoint_name = get_checkpoint_name(output_dir, iteration, **kwargs)
        return checkpoint_name

    @abstractmethod
    def save(  # type: ignore[override]
        self,
        iteration: int,
        output_dir: str = None,
        train_args: AtorchTrainingArgs = None,
        best_model_checkpoint=None,
        module=None,
        optimizer=None,
        scheduler=None,
        num_floating_point_operations_so_far=None,
        **kwargs,
    ):
        pass


class MegatronOriginSaver(MegatronCkptSaver):

    _checkpointing_context: Optional[Dict] = None

    def save(  # type: ignore[override]
        self,
        iteration: int,
        output_dir: str = None,
        train_args: AtorchTrainingArgs = None,
        best_model_checkpoint=None,
        module=None,
        optimizer=None,
        scheduler=None,
        num_floating_point_operations_so_far=None,
        **kwargs,
    ):

        megatron_args = get_args()

        if output_dir is not None:
            megatron_args.save = output_dir

        if megatron_args.ckpt_assume_constant_structure and MegatronOriginSaver._checkpointing_context is None:
            MegatronOriginSaver._checkpointing_context = {}

        if is_main_process():
            checkpoint_dir_path = self.get_interation_path(output_dir, iteration, return_base_dir=True)
            os.makedirs(checkpoint_dir_path, exist_ok=True)
            move_not_empty_iteration_path_as_backup(
                Path(self.get_interation_path(output_dir=megatron_args.save, iteration=iteration))
            )

        torch.distributed.barrier()

        from megatron.training.checkpointing import save_checkpoint

        extra_args: Dict[str, Any] = {}

        custom_async_finalize_fn: Callable = (
            kwargs["custom_async_finalize_fn"] if "custom_async_finalize_fn" in kwargs else None
        )
        if custom_async_finalize_fn is not None:
            extra_args.update(custom_async_finalize_fn=custom_async_finalize_fn)

        save_checkpoint(
            iteration,
            module,
            optimizer,
            scheduler,
            num_floating_point_operations_so_far=num_floating_point_operations_so_far,
            checkpointing_context=MegatronOriginSaver._checkpointing_context,
            **extra_args,
        )

        torch.distributed.barrier()

        if is_main_process() and train_args.save_total_limit is not None:
            from atorch.trainer.base.checkpoint import _rotate_checkpoints

            _rotate_checkpoints(
                output_dir=megatron_args.save,
                save_total_limit=train_args.save_total_limit,
                best_model_checkpoint=best_model_checkpoint,
            )

        torch.distributed.barrier()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        maybe_finalize_async_save(False)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        maybe_finalize_async_save(True)
