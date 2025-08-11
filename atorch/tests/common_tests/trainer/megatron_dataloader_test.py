import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.common.util_func import find_free_port
from atorch.utils.import_util import is_megatron_lm_available, torch_version

pytestmark = pytest.mark.core24
pytest.importorskip("torch", minversion="2.0.9")

python_version = sys.version_info


if is_megatron_lm_available():
    from megatron.core import parallel_state
    from megatron.training.arguments import parse_args
    from megatron.training.global_vars import set_args

    from atorch.trainer.megatron.megatron_dataloader import (
        MegatronDataloaderWrapper,
        skip_first_batches_for_megatron_dataloader,
    )
else:
    pytest.skip("megatron not available.", allow_module_level=True)


class DummyDataset(Dataset):
    def __init__(self, size=100, max_words=30):
        self.size = size
        self.max_words = max_words

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.ones([self.max_words], dtype=torch.int64)
        labels = torch.ones([self.max_words], dtype=torch.int64)
        attention_mask = torch.ones([self.max_words], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def create_test_args():
    args = parse_args(ignore_unknown_args=True)
    args.micro_batch_size = 2
    args.global_batch_size = 16
    args.data_parallel_size = atorch.world_size()

    return args


def _test_wrap_megatron_dataloader(rank, test_args):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not atorch.init_distributed(backend, set_cuda_device_using_local_rank=True):
        raise Exception("init failed")

    args = create_test_args()
    set_args(args)

    args.rank = rank
    args.world_size = atorch.world_size()

    parallel_state._MODEL_PARALLEL_GROUP = torch.distributed.new_group(backend=backend)

    if test_args.vpp_size == 0:
        args.virtual_pipeline_model_parallel_size = None
        dummy_dataset = DummyDataset(size=1000)
        train_sampler = DistributedSampler(dummy_dataset, num_replicas=args.world_size, rank=args.rank)
        dataloader = DataLoader(train_sampler, batch_size=args.micro_batch_size, sampler=train_sampler)

        megatron_dataloader = MegatronDataloaderWrapper(dataloader, is_post_training=True)
    else:
        args.virtual_pipeline_model_parallel_size = test_args.vpp_size

        dataloader = [None for _ in range(args.virtual_pipeline_model_parallel_size)]

        if args.rank in [0, args.world_size - 1]:
            dummy_dataset = DummyDataset(size=1000)
            train_sampler = DistributedSampler(dummy_dataset, num_replicas=args.world_size, rank=args.rank)
            real_dataloader = DataLoader(train_sampler, batch_size=args.micro_batch_size, sampler=train_sampler)

            dataloader[0 if args.rank == 0 else args.virtual_pipeline_model_parallel_size - 1] = real_dataloader

        megatron_dataloader = MegatronDataloaderWrapper(dataloader, is_post_training=True)

    megatron_dataloader = skip_first_batches_for_megatron_dataloader(megatron_dataloader, num_batches=10)

    assert isinstance(
        megatron_dataloader, MegatronDataloaderWrapper
    ), f"'megatron_dataloader' should be MegatronDataloaderWrapper type, but got {type(megatron_dataloader)}."

    atorch.reset_distributed()


class TestMegatronDataloader:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
    @pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")  # type: ignore
    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="run with cpu or gpu_num >=4")
    @pytest.mark.skipif(
        not (python_version.major >= 3 and python_version.minor >= 10), reason="Megatron 0.11 requires python >= 3.10"
    )
    @pytest.mark.parametrize("vpp_size", [0, 4])
    def test_wrap_megatron_dataloader(self, vpp_size):
        world_size = 4

        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        test_args = SimpleNamespace()
        test_args.vpp_size = vpp_size

        mp.spawn(
            _test_wrap_megatron_dataloader,
            args=(test_args,),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )

        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
