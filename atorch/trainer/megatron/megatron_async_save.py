"""Input/output checkpointing."""
import copy
import inspect
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import wrapt
from transformers import TrainerControl, TrainerState, TrainingArguments

from atorch.common.log_utils import default_logger as logger
from atorch.common.log_utils import log_rank_0
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.dist_checkpointing.strategies.async_torch_save_strategy import AsyncTorchSaveStrategy
from atorch.trainer.base.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from atorch.trainer.megatron.megatron_ckpt_saver import MegatronCkptSaver, move_not_empty_iteration_path_as_backup
from atorch.trainer.utils import is_main_process


@wrapt.decorator
def log_entry_exit(wrapped, instance, args, kwargs):
    logger.info(f"Start to {wrapped.__name__} with args={args}, kwargs={kwargs}")
    result = wrapped(*args, **kwargs)
    logger.info(f"Finish {wrapped.__name__} with args={args}, kwargs={kwargs}, result={result}")
    return result


shutil.rmtree = log_entry_exit(shutil.rmtree)
os.removedirs = log_entry_exit(os.removedirs)
os.rmdir = log_entry_exit(os.rmdir)
os.mkdir = log_entry_exit(os.mkdir)

try:
    import megatron
    from megatron.training import get_args
    from megatron.training.checkpointing import ensure_directory_exists, get_checkpoint_tracker_filename
    from megatron.training.checkpointing import save_checkpoint as megatron_save

    native_checkpoint_filename = get_checkpoint_tracker_filename
except ImportError:
    logger.warning("Please check the megatron.training checkpointing exists.")


class CheckpointConstant:
    MODEL_STATES_NAME = "model_states"
    OPTIM_STATES_NAME = "optim_states"


_MODEL_SD_NAME = "model_optim_rng.pt"
_DIST_OPTIM_SD_NAME = "distrib_optim.pt"

torch_native_save = torch.save

_async_calls_queue = AsyncCallsQueue()


class MegatronStateDictHolder:
    def __init__(self):
        self.state_dict = {}
        self.paths = {}

    def save(self, state_dict, path: str):
        if not isinstance(path, str):
            torch_native_save(state_dict, path)
            return
        if path.endswith(_MODEL_SD_NAME):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DIST_OPTIM_SD_NAME):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        else:
            raise ValueError(
                "MegatronCheckpointer only support the path whose suffix is "
                f"{_MODEL_SD_NAME} or {_DIST_OPTIM_SD_NAME}."
            )
        self.state_dict[sd_name] = state_dict
        self.paths[sd_name] = path


class MegatronAsyncCkptSaver(MegatronCkptSaver):
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
    ):
        megatron_args = get_args()
        async_timeout = train_args.flash_checkpoint_timeout
        saver = MegatronStateDictHolder()
        sig = inspect.signature(megatron_save)
        try:
            # patch methods

            # this method will:
            # 1. get model and optimizer params from origin megatron sync save method, and put them into shm.
            # 2. start a new process to save the params from shm into disk.
            # Therefore, when invoking megatron.training.checkpointing.save_checkpoint, there will save the model and
            # optimizer params into dick, and then update the latest_checkpointed_iteration.txt file, that's what we
            # don't want.
            # In order to solve the problems, we patch the origin torch.save to get the params instead of saving them
            # on disk; we patch the megatron.training.checkpointing.get_checkpoint_tracker_filename method,  to update
            # a not load related file when the megatron origin save method "finish" the save process.

            torch.save = saver.save
            megatron.training.checkpointing.get_checkpoint_tracker_filename = get_checkpoint_tracker_sync_stage_filename

            if "num_floating_point_operations_so_far" in sig.parameters:
                megatron_save(
                    iteration,
                    module,
                    optimizer,
                    scheduler,
                    num_floating_point_operations_so_far,
                )
            else:
                megatron_save(iteration, module, optimizer, scheduler)
        finally:
            torch.save = torch_native_save
            megatron.training.checkpointing.get_checkpoint_tracker_filename = native_checkpoint_filename

        # copy to shared memory
        save_fn_args = [
            (
                recursive_to_cpu(saver.state_dict.get(CheckpointConstant.MODEL_STATES_NAME, {})),
                saver.paths.get(CheckpointConstant.MODEL_STATES_NAME, "/dummy"),
            ),
            (
                saver.state_dict.get(CheckpointConstant.OPTIM_STATES_NAME, {}),
                saver.paths.get(CheckpointConstant.OPTIM_STATES_NAME, "/dummy"),
            ),
        ]

        save_strategy = AsyncTorchSaveStrategy("torch_dist", 1)
        async_save_request = async_normal_save(save_fn_args, save_strategy)  # type: ignore[arg-type]

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(megatron_args.save)
            ensure_directory_exists(tracker_filename)

            move_not_empty_iteration_path_as_backup(
                Path(self.get_interation_path(output_dir=megatron_args.save, iteration=iteration))
            )

            def iter_finalize_fn():
                with open(tracker_filename, "w") as f:
                    f.write(str(iteration))
                logger.info(
                    "successfully async saved checkpoint from iteration {:7d} to {}".format(
                        iteration, megatron_args.save
                    )
                )
                if megatron_args.log_progress:
                    from megatron.training.training import append_to_progress_log

                    append_to_progress_log(f"Saved async checkpoint\tIteration: {iteration}")

            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)

        schedule_async_save(async_save_request, timeout=async_timeout)

        log_rank_0(
            "  scheduled an async checkpoint save at iteration {:7d} to {}".format(iteration, megatron_args.save)
        )

        if is_main_process() and train_args.save_total_limit is not None:
            from atorch.trainer.base.checkpoint import _rotate_checkpoints

            _rotate_checkpoints(
                output_dir=megatron_args.save,
                save_total_limit=train_args.save_total_limit,
                best_model_checkpoint=best_model_checkpoint,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        maybe_finalize_async_save(False)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        maybe_finalize_async_save(True)


def get_checkpoint_tracker_sync_stage_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, "latest_checkpointed_sync_stage_iteration.txt")


def schedule_async_save(async_request: AsyncRequest, timeout=20 * 60):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request, timeout=timeout)


def maybe_finalize_async_save(blocking: bool = False):
    """Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
    """

    if blocking and _async_calls_queue.get_num_unfinalized_calls() > 0:
        log_rank_0("Unfinalized async checkpoint saves. Finalizing them synchronously now.")

    _async_calls_queue.maybe_finalize_async_calls(blocking)


def recursive_to_cpu(obj):
    """
    tensors will be moved to shm. Only used for async save
    """
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cpu(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_cpu(item) for item in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to("cpu").share_memory_()
    else:
        return copy.deepcopy(obj)


def async_normal_save(
    save_fn_args: List[Tuple],
    sharded_strategy: AsyncTorchSaveStrategy = None,
) -> Optional[AsyncRequest]:
    assert isinstance(sharded_strategy, AsyncTorchSaveStrategy)
    async_request = sharded_strategy.async_save(save_fn_args)
    return async_request
