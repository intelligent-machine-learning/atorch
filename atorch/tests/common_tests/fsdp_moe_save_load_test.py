# flake8: noqa: E402
import os
import shutil
import unittest

import pytest
import torch

from atorch.utils.version import torch_version

if not torch.cuda.is_available() or torch_version() >= (2, 5, 0):  # type: ignore
    pytest.skip("requires cuda device with torch 2.1 or 2.4", allow_module_level=True)

pytestmark = pytest.mark.core24

import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import CustomPolicy

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import divide, find_free_port
from atorch.distributed.distributed import create_parallel_group, parallel_group
from atorch.tests.toy_modules.toy_for_moe import (
    DummyBlock,
    DummyExperts,
    assert_same_sd,
    get_input,
    get_model,
    get_optim,
    optim_param_func,
)
from atorch.tests.toy_modules.toy_for_moe import run_module_gt as run_module_gt_toy
from atorch.utils.fsdp_init_util import clear_fsdp_patch_init, patch_fsdp_init
from atorch.utils.fsdp_save_util import ShardOptim, save_fsdp_flat_param, save_fsdp_optim_param


def init_moe_group():
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()
    ep_size = divide(world_size, 2)
    fsdp_mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(fsdp_mode)
    ep_mode = ([("expert", ep_size), ("expert_fsdp", 2)], None)
    create_parallel_group(ep_mode)


def moe_fsdp_policy_fn(module):
    if isinstance(module, DummyBlock):
        # non experts fsdp wrap
        return {"process_group": parallel_group("data")}
        # return True
    elif isinstance(module, DummyExperts):
        # experts fsdp wrap
        return {"process_group": parallel_group("expert_fsdp")}
    return False


def run_module_gt(rank, world_size, hidden_size, path):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()
    m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_fsdp_policy_fn),
    }
    model = FSDP(m, **fsdp_config)
    optim = get_optim(model)
    loss = model(get_input(hidden_size)).mean()
    loss.backward()
    optim.step()
    save_fsdp_flat_param(model, path)
    save_fsdp_optim_param(model, optim, path)


def run_module_load(rank, world_size, hidden_size, gt_ckpt):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()

    # ref_model
    ref_m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_fsdp_policy_fn),
    }
    ref_model = FSDP(ref_m, **fsdp_config)
    ref_optim = get_optim(ref_model)
    loss = ref_model(get_input(hidden_size)).mean()
    loss.backward()
    ref_optim.step()

    # load_model
    load_m = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_m,
    )
    # patch fsdp init load need sync_module_states to be False and to_empty param_init_fn
    fsdp_config["sync_module_states"] = False
    fsdp_config["param_init_fn"] = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    load_model = FSDP(load_m, **fsdp_config)  # ckpt load already done here
    load_optim = get_optim(load_model)
    sm = ShardOptim(gt_ckpt)
    reshard_optim_state = sm.reshard_optim_state_dict(load_model)
    load_optim.load_state_dict(reshard_optim_state)
    assert_same_sd(ref_model.state_dict(), load_model.state_dict())
    assert_same_sd(ref_optim.state_dict(), load_optim.state_dict())


def run_module_double_load(rank, world_size, hidden_size, gt_ckpt, use_cpu_offload=False):
    # ref_model
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_fsdp_policy_fn),
    }

    if use_cpu_offload:
        fsdp_config["cpu_offload"] = True
    # patch fsdp init load need sync_module_states to be False and to_empty param_init_fn
    fsdp_config["sync_module_states"] = False
    fsdp_config["param_init_fn"] = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    strategy = [("fsdp", fsdp_config)]

    # load_model
    load_model1 = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_model1,
    )

    _, result1, _ = auto_accelerate(
        load_model1,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model1 = result1.model
    optim1 = result1.optim
    print("load_model1:", model1.sharding_strategy)

    # clear patch
    clear_fsdp_patch_init()

    load_model2 = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_model2,
    )
    _, result2, _ = auto_accelerate(
        load_model2,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model2 = result2.model
    optim2 = result2.optim
    print("load_model2:", model2.sharding_strategy)

    # train
    input_data = get_input(hidden_size)
    loss1 = model1(input_data).mean()
    loss1.backward()
    optim1.step()

    loss2 = model2(input_data).mean()
    loss2.backward()
    optim2.step()


class FSDPMoEShardSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPMoEShardSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.test_dir_prefix = str(self.world_size)
        self.gt_ckpt = f"/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}/gt"
        if os.path.exists("/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}"):
            shutil.rmtree("/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}")
        self._prepare_toy_save()

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Must have at least 4 GPUs for gpu test",
    )
    @pytest.mark.core24
    def test_toy_load(self):
        """This test will load toy module"""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    "Must have at least 4 GPUs for gpu test",
)
class FSDPMoEDPOTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPMoEDPOTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.gt_ckpt = f"/tmp/fsdp_moe_dpo_test/gt"
        self._prepare_toy_save()

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""

        if os.path.exists("/tmp/fsdp_moe_dpo_test/"):
            shutil.rmtree("/tmp/fsdp_moe_dpo_test/")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt_toy,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp_double_init_and_accerate(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_double_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp_cpu_offload(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_double_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()
