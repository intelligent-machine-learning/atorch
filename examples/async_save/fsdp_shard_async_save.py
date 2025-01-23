import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

try:
    from torch.distributed.checkpoint.stateful import Stateful  # noqa: F401
except Exception:
    raise ImportError("To use FSDP dcp save/load, you need pytorch version >= 2.4.0")

from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

CHECKPOINT_DIR = "/tmp/async_shard_save"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    import torch

    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    # loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    try:
        from atorch.trainer.args import DummyAtorchTrainingArgs
        from atorch.trainer.fsdp.fsdp_ckpt_saver import ExtraState
        from atorch.trainer.fsdp.fsdp_shard_async_saver import FsdpShardCkptAsyncSaver

        saver = FsdpShardCkptAsyncSaver()
        extraState = ExtraState({"testK1": "testV1", "testK2": "testV2"})

        saver.save(
            iteration=5,  # your saving step
            output_dir=CHECKPOINT_DIR,  # e.g. /tmp/llama_7b_test
            train_args=DummyAtorchTrainingArgs(save_total_limit=3),
            module=model,
            optimizer=optimizer,
            extra_state=extraState,
        )
    except Exception:
        print(f"some exception happens on rank: {rank}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
