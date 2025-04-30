import os.path
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

from atorch.common.log_utils import default_logger as logger
from atorch.common.log_utils import log_rank_0
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.fsdp.fsdp_ckpt_loader import FsdpCkptLoader
from atorch.trainer.fsdp.fsdp_ckpt_saver import ExtraState, get_ckpt_trace_file_path
from atorch.trainer.fsdp.fsdp_shard_async_saver import FsdpShardDcpCkpt


def get_ckpt_meta(trace_filename):
    iteration = 0
    with open(trace_filename, "r") as f:
        meta_str = f.read().strip()
        try:
            iteration = int(meta_str)
        except ValueError:
            log_rank_0("ERROR: Invalid metadata file {}. Exiting".format(trace_filename))
            sys.exit()
    assert iteration > 0, "error parsing metadata file {}".format(trace_filename)

    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        max_iter = iters_cuda[0].item()

        if iteration != max_iter:
            rank = torch.distributed.get_rank()
            logger.warning(
                f"WARNING: on rank {rank} found iteration {iteration} in the metadata while max iteration "
                f"across the ranks is {max_iter}. Replacing it with max iteration. Please make sure all "
                f"ranks are loading the same ckpt, and not saving a new ckpt during loading."
            )
    else:
        max_iter = iteration
    return max_iter


class FsdpShardDcpCkptLoader(FsdpCkptLoader):
    def load(
        self,
        resume_from_ckpt,
        model,
        train_args: AtorchTrainingArgs = None,
        optimizer=None,
        extra_state: ExtraState = None,
        ckpt_step=None,
        **kwargs,
    ) -> int:
        if not resume_from_ckpt:
            log_rank_0("No sharded_state_dict checkpoint directory found")
            raise ValueError(
                f"cannot find folder path: {resume_from_ckpt}, fail to load model and/or optimizer,"
                f"please make sure you pass the correct ckpt path in."
            )

        if ckpt_step is not None:
            iteration = ckpt_step
        else:
            tracker_filename = get_ckpt_trace_file_path(resume_from_ckpt)
            iteration = get_ckpt_meta(tracker_filename)

        ckpt_path = os.path.join(resume_from_ckpt, "iter_{:07d}".format(iteration))

        reader = FileSystemReader(ckpt_path)

        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(), ShardedOptimStateDictConfig()
        ):
            state_dict = {
                "atorch_FsdpShardDcpCkpt": FsdpShardDcpCkpt(
                    model=model,
                    optimizer=optimizer,
                )
            }

            if extra_state is not None:
                extra_state.load_state(file_path=extra_state.get_save_filepath(output_dir=ckpt_path))
            dcp.load(state_dict=state_dict, storage_reader=reader)

        log_rank_0(f"Load ckpt from iteration {iteration} success!")

        return iteration
