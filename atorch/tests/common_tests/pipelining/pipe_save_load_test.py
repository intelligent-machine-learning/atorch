import functools
import os
import shutil
import unittest

import pytest
import torch
import torch.multiprocessing as mp
from model_registry import get_llama_model_chunk, llama_loss_func
from transformers.models.llama.configuration_llama import LlamaConfig

import atorch
from atorch.checkpoint.torch_checkpoint import TorchDCPCheckpointManager, TrainState
from atorch.common.util_func import data_to_device, find_free_port
from atorch.distributed.distributed import create_parallel_group
from atorch.pipeline_parallel.pipe_module import PipeModuleConfig, make_pipe_module
from atorch.pipeline_parallel.pipe_schedule import make_pipe_schedule_from_pipe_module
from atorch.utils.version import torch_version

pytestmark = pytest.mark.core24

skip = None
if torch_version() >= (2, 4, 0):  # type: ignore
    skip = False
else:
    skip = True


hidden_size = 256
head_num = 4
key_value_head_num = 4
seq_length = 128
intermediate_size = 256
microbatch_num = 8


class OptimizersContainer:
    """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()


def pre_create_group(pp_size):
    world_size = torch.distributed.get_world_size()
    dp_size = world_size // pp_size
    if dp_size > 1:
        config = ([("data", dp_size), ("pipe", pp_size)], None)
    else:
        config = ([("pipe", pp_size)], None)
    create_parallel_group(config)


def get_input(vocab_size, seed=123, length=1024):
    diff_seed = seed + torch.distributed.get_rank()
    torch.cuda.manual_seed(diff_seed)
    torch.manual_seed(diff_seed)
    samples = [torch.randint(1, vocab_size, (length,)) for _ in range(8)]
    input_ids = torch.stack(samples, 0)
    labels = input_ids.clone()
    batch = {"input_ids": input_ids, "labels": labels}
    return batch


def gen_model_config(layer_num):
    model_config = LlamaConfig()
    c_s = f"hidden_size={hidden_size},num_attention_heads={head_num},num_hidden_layers={layer_num},"
    c_s += f"num_key_value_heads={key_value_head_num},max_position_embeddings={seq_length},"
    c_s += f"intermediate_size={intermediate_size}"
    model_config.update_from_string(c_s)
    model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = False

    return model_config


def create_model_and_pp_sche(model_config, layer_num, pp_stage_num, pp_num, interleaved=False):
    activation_mapping = [("hidden_states", 0)]
    batch_mapping = [("input_ids", 0)]
    default_input_info = (activation_mapping, None)
    stage_0_input_info = (None, batch_mapping)
    io_mapping = {"default": default_input_info, 0: stage_0_input_info}

    # create PipeModuleConfig
    sche_name = "ScheduleInterleaved1F1B" if interleaved else "Schedule1F1B"
    pm_config = PipeModuleConfig(
        model_config=model_config,
        total_layer_num=layer_num,
        tie_weight_info=None,
        input_output_mapping=io_mapping,
        virtual_pp_size=pp_stage_num // pp_num,
        auto_ddp=True,
        sche_name=sche_name,
        n_microbatches=microbatch_num,
    )

    # create pipe_module
    # use amp by default
    strategy = []
    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy.append(("amp_native", amp_config))
    loss_fn = functools.partial(llama_loss_func, vocab_size=model_config.vocab_size)
    pipe_module = make_pipe_module(
        model_provider=get_llama_model_chunk, loss_func=loss_fn, strategy=strategy, config=pm_config
    )

    # create schedule
    schedule = make_pipe_schedule_from_pipe_module(pipe_module, pm_config)

    optim_func = torch.optim.AdamW
    optim_args = {"lr": 0.001}
    optimizer = optim_func(pipe_module.parameters(), **optim_args)
    return schedule, [pipe_module], [optimizer], [None]


def run_pp_save(rank, world_size, save_load_folder, interleaved=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    if interleaved:
        pp_num = world_size
        pp_stage_num = world_size * 2
        layer_num = world_size * 2
    else:
        pp_num = pp_stage_num = world_size
        layer_num = world_size

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    pre_create_group(pp_size=pp_num)

    pp_rank = atorch.distributed.distributed.parallel_rank("pipe")
    pp_size = atorch.distributed.distributed.parallel_group_size("pipe")
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1

    model_config = gen_model_config(layer_num)

    schedule, model_parts, optimizers, lr_schedules = create_model_and_pp_sche(
        model_config, layer_num, pp_stage_num, pp_num, interleaved
    )
    optim = OptimizersContainer(optimizers)
    device = atorch.local_rank()
    train_state = TrainState()

    checkpoint = TorchDCPCheckpointManager(
        config={"save_load_folder": save_load_folder},
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedules,
        states={"train_state": train_state},
    )

    loop_num = 1
    for _ in range(loop_num):
        train_state.step += 1
        optim.zero_grad()
        batch = get_input(vocab_size=model_config.vocab_size)
        batch = data_to_device(batch, device=device)
        if is_first_stage:
            schedule.step(batch["input_ids"])
        elif is_last_stage:
            schedule.step(target=batch["labels"])
        else:
            schedule.step()
        optim.step()

        if train_state.step == loop_num:
            checkpoint.save(train_state.step)

    step_folder = os.path.join(save_load_folder, f"step-{loop_num}")
    assert os.path.exists(step_folder)
    assert len(os.listdir(step_folder)) == 5
    metadata_path = os.path.join(step_folder, ".metadata")
    assert os.path.exists(metadata_path) and os.path.isfile(metadata_path)


def run_pp_load(rank, world_size, save_load_folder, interleaved=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    if interleaved:
        pp_num = world_size
        pp_stage_num = world_size * 2
        layer_num = world_size * 2
    else:
        pp_num = pp_stage_num = world_size
        layer_num = world_size

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    pre_create_group(pp_size=pp_num)

    pp_rank = atorch.distributed.distributed.parallel_rank("pipe")
    pp_size = atorch.distributed.distributed.parallel_group_size("pipe")
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1

    model_config = gen_model_config(layer_num)

    schedule1, model_parts1, optimizers1, lr_schedules1 = create_model_and_pp_sche(
        model_config, layer_num, pp_stage_num, pp_num, interleaved
    )
    optim1 = OptimizersContainer(optimizers1)
    device = atorch.local_rank()
    train_state1 = TrainState()

    checkpoint = TorchDCPCheckpointManager(
        config={"save_load_folder": save_load_folder},
        model_parts=model_parts1,
        optimizers=optimizers1,
        lr_schedulers=lr_schedules1,
        states={"train_state": train_state1},
    )
    checkpoint.load(step=1)

    # prepare ref model
    schedule2, _, optimizers2, _ = create_model_and_pp_sche(model_config, layer_num, pp_stage_num, pp_num, interleaved)
    optim2 = OptimizersContainer(optimizers2)

    for _ in range(1):
        optim2.zero_grad()
        batch = get_input(vocab_size=model_config.vocab_size)
        batch = data_to_device(batch, device=device)
        if is_first_stage:
            schedule2.step(batch["input_ids"])
        elif is_last_stage:
            schedule2.step(target=batch["labels"])
        else:
            schedule2.step()
        optim2.step()

    for _ in range(2):
        optim1.zero_grad()
        batch1 = get_input(vocab_size=model_config.vocab_size)
        batch1 = data_to_device(batch1, device=device)

        if is_first_stage:
            schedule1.step(batch1["input_ids"])
        elif is_last_stage:
            losses1 = []
            schedule1.step(target=batch1["labels"], losses=losses1)
        else:
            schedule1.step()
        optim1.step()

        optim2.zero_grad()
        batch2 = {k: batch1[k].detach().clone() for k in batch1}
        if is_first_stage:
            schedule2.step(batch2["input_ids"])
        elif is_last_stage:
            losses2 = []
            schedule2.step(target=batch2["labels"], losses=losses2)
        else:
            schedule2.step()
        optim2.step()

    if is_last_stage:
        assert torch.allclose(sum(losses1), sum(losses2), rtol=1e-04)


@unittest.skipIf(torch.cuda.device_count() < 4 or skip, "Requires 4 gpus.")
class PipelineSaveLoadTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.world_size = 4
        self.save_load_folder = "/tmp/pipe_save_load_test/"

    def _save(self, interleaved=False):
        if os.path.exists(self.save_load_folder):
            shutil.rmtree(self.save_load_folder)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_pp_save,
            nprocs=self.world_size,
            args=(self.world_size, self.save_load_folder, interleaved),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    @unittest.skipIf(True, "To reduce time")
    def test_save(self):
        self._save()

    @unittest.skipIf(True, "To reduce time")
    def test_save_interleaved(self):
        self._save(interleaved=True)

    def test_load(self):
        self._save()

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_pp_load,
            nprocs=self.world_size,
            args=(self.world_size, self.save_load_folder),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_load_interleaved(self):
        self._save(interleaved=True)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_pp_load,
            nprocs=self.world_size,
            args=(self.world_size, self.save_load_folder, True),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()
