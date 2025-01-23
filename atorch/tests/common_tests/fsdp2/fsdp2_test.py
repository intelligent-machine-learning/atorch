# flake8: noqa: E402
import os
import shutil
import unittest
from functools import partial
from pathlib import Path

import pytest
import torch

import atorch
from atorch.utils.version import torch_version

if not torch.cuda.is_available():
    pytest.skip("requires cuda device", allow_module_level=True)

is_torch_bigger_than_25 = False
if torch_version() >= (2, 5, 0):  # type: ignore
    is_torch_bigger_than_25 = True
else:
    is_torch_bigger_than_25 = False

pytestmark = pytest.mark.core24


import torch.multiprocessing as mp
from torch.distributed.fsdp.wrap import CustomPolicy

from atorch.auto import auto_accelerate
from atorch.checkpoint.torch_checkpoint import TorchDCPCheckpointManager, TrainState
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import get_device_mesh
from atorch.tests.toy_modules.toy_for_moe import (
    DummyBlock,
    DummyExperts,
    get_input,
    get_model,
    init_moe_group,
    moe_hsdp_policy_fn,
    optim_param_func,
)


def moe_hsdp_policy_fn_device_mesh(module, nested_fsdp2=False):
    data_device_mesh = get_device_mesh("data")
    if nested_fsdp2:
        expert_fsdp_device_mesh = get_device_mesh("expert_fsdp")
        cls_to_fsdp_device_mesh = {DummyBlock: data_device_mesh, DummyExperts: expert_fsdp_device_mesh}
    else:
        cls_to_fsdp_device_mesh = {DummyBlock: data_device_mesh}

    if module.__class__ in cls_to_fsdp_device_mesh:
        dm = cls_to_fsdp_device_mesh[module.__class__]
        return {"device_mesh": dm}
    return False


def param_init_fn(module, initializer_range=0.02):
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()


def loop_and_check(model, name_to_checkpoint_name):
    buffer_name_to_checkpoint_name = name_to_checkpoint_name[0]
    param_name_to_checkpoint_name = name_to_checkpoint_name[1]
    for name, buffer in model.named_buffers():
        checkpoint_name = buffer_name_to_checkpoint_name[name]
        if Path(checkpoint_name).exists():
            loaded_buffer = torch.load(checkpoint_name)
            if loaded_buffer.dtype is not buffer.dtype:
                loaded_buffer = loaded_buffer.to(buffer.dtype)
            if loaded_buffer.device is not buffer.device:
                loaded_buffer = loaded_buffer.to(buffer.device)
            assert torch.allclose(buffer, loaded_buffer)

    for name, param in model.named_parameters():
        checkpoint_name = param_name_to_checkpoint_name[name]
        if Path(checkpoint_name).exists():
            loaded_param = torch.load(checkpoint_name)
            if loaded_param.dtype is not param.dtype:
                loaded_param = loaded_param.to(param.dtype)
            if loaded_param.device is not param.device:
                loaded_param = loaded_param.to(param.device)
            assert torch.allclose(param.full_tensor(), loaded_param)


def run_module_fsdp2(
    rank,
    world_size,
    hidden_size,
    use_atorch_wrap_cls=False,
    use_parallel_mode=False,
    use_meta_init=False,
    use_param_init_fn=False,
    offload_disk=False,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    if not use_parallel_mode:
        init_moe_group(use_device_mesh=True, ep_size=1)
    else:
        atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)

    if use_atorch_wrap_cls:
        fsdp_config = {
            "atorch_wrap_cls": (DummyBlock,),
        }
    else:
        fsdp_config = {
            "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn_device_mesh),
        }
    if use_param_init_fn:
        fsdp_config["param_init_fn"] = param_init_fn

    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}

    parallel_mode = ([("data", torch.distributed.get_world_size())], None, False, True)
    strategy = [("parallel_mode", parallel_mode)] if use_parallel_mode else []
    strategy.extend([("fsdp2", fsdp_config), ("amp_native", amp_config)])

    # load_model
    model = get_model(hidden_size, meta=use_meta_init, offload_disk=offload_disk)

    if offload_disk:
        buffer_name_to_checkpoint_name = {name: buffer.checkpoint_name for name, buffer in model.named_buffers()}
        param_name_to_checkpoint_name = {name: param.checkpoint_name for name, param in model.named_parameters()}
        name_to_checkpoint_name = [buffer_name_to_checkpoint_name, param_name_to_checkpoint_name]

    _, result1, _ = auto_accelerate(
        model,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model1 = result1.model
    optim1 = result1.optim

    if offload_disk:
        loop_and_check(model1, name_to_checkpoint_name)

    if use_meta_init and not use_param_init_fn and not offload_disk:
        with torch.no_grad():
            for _, param in model1.named_parameters():
                assert param.sum().item() == 0

    # train
    input_data = get_input(hidden_size)
    loss1 = model1(input_data).mean()
    loss1.backward()
    optim1.step()

    optim1.zero_grad()
    if offload_disk:
        for _, param in model1.named_parameters():
            assert param.grad is None


def run_moe_nested_fsdp2(
    rank,
    world_size,
    hidden_size,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(use_device_mesh=True, ep_size=1)
    nested_fsdp2_policy = partial(moe_hsdp_policy_fn_device_mesh, nested_fsdp2=True)
    fsdp_config = {
        "auto_wrap_policy": CustomPolicy(nested_fsdp2_policy),
    }
    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy = [("fsdp2", fsdp_config), ("amp_native", amp_config)]
    # load_model
    model1 = get_model(hidden_size, meta=False, offload_disk=False)

    _, result1, _ = auto_accelerate(
        model1,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        # Note: foreach should be False when nested fsdp2 is used
        optim_args={"lr": 0.001, "foreach": False},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model1 = result1.model
    optim1 = result1.optim

    # ref model
    fsdp1_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn),
    }
    strategy2 = [("fsdp", fsdp1_config), ("amp_native", amp_config)]
    model2 = get_model(hidden_size, meta=False, offload_disk=False)
    _, result2, _ = auto_accelerate(
        model2,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        # Note: foreach should be False when nested fsdp2 is used
        optim_args={"lr": 0.001, "foreach": False},
        load_strategy=strategy2,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model2 = result2.model
    optim2 = result2.optim

    # train
    for _ in range(10):
        optim1.zero_grad()
        optim2.zero_grad()

        input_data = get_input(hidden_size)
        loss1 = model1(input_data).mean()
        loss1.backward()
        optim1.step()

        # train
        input_data_2 = input_data.detach().clone()
        loss2 = model2(input_data_2).mean()
        loss2.backward()
        optim2.step()

        torch.allclose(loss1, loss2)


def _create_and_accelerate_model(hidden_size, nested_fsdp2=False):
    if nested_fsdp2:
        nested_fsdp2_policy = partial(moe_hsdp_policy_fn_device_mesh, nested_fsdp2=True)
        fsdp_config = {
            "auto_wrap_policy": CustomPolicy(nested_fsdp2_policy),
        }
    else:
        fsdp_config = {
            "atorch_wrap_cls": (DummyBlock,),
        }
    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy = [("fsdp2", fsdp_config), ("amp_native", amp_config)]

    # create model
    model1 = get_model(hidden_size, meta=False, offload_disk=False)

    _, result1, _ = auto_accelerate(
        model1,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        # Note: foreach should be False when nested fsdp2 is used
        optim_args={"lr": 0.001, "foreach": False},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    return result1


def run_module_fsdp2_save(rank, world_size, hidden_size, save_load_folder, nested_fsdp2=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(use_device_mesh=True, ep_size=1)

    result1 = _create_and_accelerate_model(hidden_size, nested_fsdp2=nested_fsdp2)
    model1 = result1.model
    optim1 = result1.optim
    lr_scheduler1 = result1.lr_scheduler
    train_state = TrainState()

    checkpoint = TorchDCPCheckpointManager(
        config={"save_load_folder": save_load_folder},
        model_parts=[model1],
        optimizers=[optim1],
        lr_schedulers=[lr_scheduler1],
        states={"train_state": train_state},
    )

    loop_num = 1
    for _ in range(loop_num):
        train_state.step += 1
        optim1.zero_grad()

        input_data = get_input(hidden_size)
        loss1 = model1(input_data).mean()
        loss1.backward()
        optim1.step()

        if train_state.step == 1:
            checkpoint.save(train_state.step)

    step_folder = os.path.join(save_load_folder, f"step-{loop_num}")
    assert os.path.exists(step_folder)
    assert len(os.listdir(step_folder)) == 5
    metadata_path = os.path.join(step_folder, ".metadata")
    assert os.path.exists(metadata_path) and os.path.isfile(metadata_path)


def run_module_fsdp2_load(rank, world_size, hidden_size, save_load_folder, nested_fsdp2=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(use_device_mesh=True, ep_size=1)

    # load model trained one step already
    result1 = _create_and_accelerate_model(hidden_size, nested_fsdp2=nested_fsdp2)
    model1 = result1.model
    optim1 = result1.optim
    lr_scheduler1 = result1.lr_scheduler
    train_state1 = TrainState()

    checkpoint = TorchDCPCheckpointManager(
        config={"save_load_folder": save_load_folder},
        model_parts=[model1],
        optimizers=[optim1],
        lr_schedulers=[lr_scheduler1],
        states={"train_state": train_state1},
    )
    checkpoint.load(step=1)

    # prepare ref model
    result2 = _create_and_accelerate_model(hidden_size)
    model2 = result2.model
    optim2 = result2.optim

    input_data = get_input(hidden_size)
    loss2 = model2(input_data).mean()
    loss2.backward()
    optim2.step()

    # train both
    for _ in range(2):
        optim1.zero_grad()
        optim2.zero_grad()

        input_data = get_input(hidden_size)
        loss1 = model1(input_data).mean()
        loss1.backward()
        optim1.step()

        # train
        input_data_2 = input_data.detach().clone()
        loss2 = model2(input_data_2).mean()
        loss2.backward()
        optim2.step()

        torch.allclose(loss1, loss2)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not is_torch_bigger_than_25,
    "Must have at least 4 GPUs for gpu test",
)
class FSDP2Test(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDP2Test, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64

    def setUp(self):
        atorch.reset_distributed()

    def test_fsdp2_auto_wrap_policy(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2_atorch_wrap_cls(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2_atorch_parallel_mode(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2_meta_init_param_init_fn(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True, True, True, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2_meta_init_no_param_init_fn(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True, True, True, False, False),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2_meta_init_offload_disk(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True, True, True, False, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_moe_nested_fsdp2(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_moe_nested_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not is_torch_bigger_than_25,
    "Must have at least 4 GPUs for gpu test",
)
class FSDP2SaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDP2SaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.save_load_folder = "/tmp/fsdp2_save_load_test/"

    def _save(self, nested_fsdp2=False):
        if os.path.exists(self.save_load_folder):
            shutil.rmtree(self.save_load_folder)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2_save,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.save_load_folder, nested_fsdp2),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def _load(self, nested_fsdp2=False):
        self._save()

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.save_load_folder, nested_fsdp2),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_save(self):
        self._save()

    def test_load(self):
        self._load()

    def test_nested_fsdp2_save(self):
        self._save(nested_fsdp2=True)

    def test_nested_fsdp2_load(self):
        self._load(nested_fsdp2=True)
