# flake8: noqa: E402
import os
import unittest

import pytest
import torch
import torch.multiprocessing as mp

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

from atorch.auto import auto_accelerate
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group
from atorch.tests.toy_modules.toy_for_moe import get_input, optim_param_func
from atorch.utils.moe_util import compute_grad_norm_


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mw1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.mw2 = torch.nn.Linear(hidden_size * 2, hidden_size)

        inv_freq = torch.ones((2, 2))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        x = self.mw2(self.mw1(x))
        return x


def run_compute_grad_norm(
    rank,
    world_size,
    hidden_size,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    fsdp_mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(fsdp_mode, use_device_mesh=True)

    fsdp_config = {
        "atorch_wrap_cls": (DummyModel,),
    }
    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy = [("fsdp2", fsdp_config), ("amp_native", amp_config)]

    model = DummyModel(hidden_size).cuda()

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

    # train
    input_data = get_input(hidden_size)
    loss1 = model1(input_data).mean()
    loss1.backward()

    grad_norm1 = compute_grad_norm_(model1.parameters())
    grad_norm2 = torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0).full_tensor()
    assert torch.allclose(grad_norm1, grad_norm2)

    optim1.step()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not is_torch_bigger_than_25,
    "Must have at least 4 GPUs for gpu test",
)
class FSDP2GradNormTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDP2GradNormTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 4

    def setUp(self):
        atorch.reset_distributed()

    def test_moe_nested_fsdp2(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_compute_grad_norm,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()
