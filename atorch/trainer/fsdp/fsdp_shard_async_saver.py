from typing import List

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.storage import WriteResult

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.fsdp.fsdp_ckpt_saver import ExtraState, FsdpCkptSaver, get_ckpt_trace_file_path
from atorch.trainer.utils import is_main_process

try:
    from torch.distributed.checkpoint.stateful import Stateful

    from atorch.trainer.fsdp.dcp_forked import async_save
except Exception:
    raise ImportError("To use FSDP dcp save/load, you need pytorch version >= 2.4.0")

import os

import torch
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from atorch.common.log_utils import default_logger as logger


class FsdpShardDcpCkpt(Stateful):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):

        if self.optimizer is not None:
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type
            # to FSDP.SHARDED_STATE_DICT
            from torch.distributed.checkpoint.state_dict import get_state_dict

            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        else:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict

            model_state_dict = get_model_state_dict(self.model)
            optimizer_state_dict = None

        state_dict_to_save = {"model": model_state_dict}

        if self.optimizer is not None:
            state_dict_to_save["optim"] = optimizer_state_dict

        return state_dict_to_save

    def load_state_dict(self, state_dict: dict):
        optim_state_dict = state_dict.get("optim", None)

        model_state_dict = state_dict.get("model", None)
        assert model_state_dict is not None, "cannot find key 'model' in checkpoint, failed to load..."

        if self.optimizer is not None and optim_state_dict is None:
            logger.warning(
                "cannot find key 'optim' in checkpoint, will not load optimizer state_dict, your "
                "optimizer state_dict will remind as the input one."
            )

        if optim_state_dict is not None and self.optimizer is not None:
            from torch.distributed.checkpoint.state_dict import set_state_dict

            set_state_dict(
                self.model, self.optimizer, model_state_dict=model_state_dict, optim_state_dict=optim_state_dict
            )
        else:
            self.model.load_state_dict(model_state_dict)


class FsdpShardCkptAsyncSaver(FsdpCkptSaver):
    def save(  # type: ignore[override]
        self,
        iteration: int,
        output_dir: str = None,
        train_args: AtorchTrainingArgs = None,
        best_model_checkpoint=None,
        module=None,
        optimizer=None,
        extra_state: ExtraState = None,
    ) -> str:
        with FSDP.state_dict_type(
            module,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            ckpt_state_dict = {
                "atorch_FsdpShardDcpCkpt": FsdpShardDcpCkpt(
                    model=module,
                    optimizer=optimizer,
                )
            }

            iteration_path = self.get_interation_path(output_dir=output_dir, iteration=iteration)
            os.makedirs(iteration_path, exist_ok=True)

            class FileWriterWithLatestCkptIter(FileSystemWriter):
                def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
                    super().finish(metadata, results)
                    logger.info(
                        "Async saving FSDF process successfully done, start to invoke update latest iteration"
                        "of latest_checkpointed_iteration.txt"
                    )
                    with open(get_ckpt_trace_file_path(self.path.parent), "w") as f:
                        f.write(str(iteration))
                    logger.info(f"Async saving FSDF process successfully on iteration:{str(iteration)}")

            fs_storage_writer = FileWriterWithLatestCkptIter(path=iteration_path)

            if torch.distributed.get_rank() == 0:
                extra_state_path = extra_state.get_save_filepath(output_dir=iteration_path)
                extra_state.save_state(file_path=extra_state_path)

            checkpoint_save_future = async_save(  # noqa: F841
                state_dict=ckpt_state_dict, storage_writer=fs_storage_writer
            )

            if is_main_process() and train_args.save_total_limit is not None:
                from atorch.trainer.base.checkpoint import _rotate_checkpoints

                _rotate_checkpoints(
                    output_dir=output_dir,
                    save_total_limit=train_args.save_total_limit,
                    best_model_checkpoint=best_model_checkpoint,
                )

            return output_dir
