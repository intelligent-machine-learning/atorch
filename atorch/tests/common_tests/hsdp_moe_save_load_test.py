# flake8: noqa: E402
import os
import shutil
import unittest

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires cuda device", allow_module_level=True)

pytestmark = pytest.mark.core24

import torch.multiprocessing as mp

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import find_free_port
from atorch.tests.toy_modules.toy_for_moe import (
    DummyBlock,
    DummyExperts,
    _init_group_and_model,
    assert_same_sd,
    get_model,
    optim_param_func,
    run_module_gt,
)
from atorch.utils.fsdp_init_util import patch_fsdp_init

# from atorch.utils.fsdp_async_ckpt_util import save_checkpoint
from atorch.utils.fsdp_save_util import ShardOptim

torch.distributed.fsdp._runtime_utils._validate_and_get_hybrid_shard_state = lambda x: None


def run_module_load(rank, world_size, hidden_size, gt_ckpt, ddp1_size=None, ddp2_size=None, check_strict=True):
    # ref_model
    ref_model, ref_optim, fsdp_config = _init_group_and_model(
        rank=rank, world_size=world_size, hidden_size=hidden_size, ddp1_size=ddp1_size, ddp2_size=ddp2_size
    )
    print("ref_model:", ref_model.sharding_strategy)

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
    strategy = [("fsdp", fsdp_config)]

    status, result, best_strategy = auto_accelerate(
        load_m,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    load_model = result.model
    print("load_model:", load_model.sharding_strategy)
    load_optim = result.optim
    sm = ShardOptim(gt_ckpt)
    reshard_optim_state = sm.reshard_optim_state_dict(load_model)
    load_optim.load_state_dict(reshard_optim_state)
    assert_same_sd(ref_model.state_dict(), load_model.state_dict(), strict=check_strict)
    assert_same_sd(ref_optim.state_dict(), load_optim.state_dict(), strict=check_strict)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    "Must have at least 8 GPUs for gpu test",
)
class HSDPMoEShardSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(HSDPMoEShardSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 8
        self.hidden_size = hidden_size or 64
        self.gt_ckpt = f"/tmp/hsdp_moe_save_load_test/gt"

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self, ddp1_size=None, ddp2_size=None, async_save=False):
        """This test will save toy module"""

        if os.path.exists("/tmp/hsdp_moe_save_load_test/"):
            shutil.rmtree("/tmp/hsdp_moe_save_load_test/")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size, async_save),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_save_fsdp(self):
        self._prepare_toy_save()

    def test_save_hsdp(self):
        self._prepare_toy_save(ddp1_size=2, ddp2_size=2)

    # def test_async_save_fsdp(self):
    #     self._prepare_toy_save(async_save=True)

    # def test_async_save_hsdp(self):
    #     self._prepare_toy_save(ddp1_size=2, ddp2_size=2, async_save=True)

    def test_hsdp_load_from_hsdp_ckpt(self, async_save=False):
        """This test will load toy module"""
        ddp1_size = 2
        ddp2_size = 2
        self._prepare_toy_save(ddp1_size, ddp2_size, async_save=async_save)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    # def test_hsdp_load_from_hsdp_async_save_ckpt(self):
    #     self.test_hsdp_load_from_hsdp_ckpt(async_save=True)

    def test_hsdp_load_from_fsdp_ckpt(self, async_save=False):
        """This test will load toy module"""
        self._prepare_toy_save(async_save=async_save)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        ddp1_size = 2
        ddp2_size = 2
        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size, False),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    # def test_hsdp_load_from_fsdp_async_save_ckpt(self):
    #     self.test_hsdp_load_from_fsdp_ckpt(async_save=True)
