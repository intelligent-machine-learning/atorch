import os

import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import CustomPolicy

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import divide
from atorch.distributed.distributed import create_parallel_group, parallel_group, parallel_group_and_ranks
from atorch.utils.fsdp_save_util import save_fsdp_flat_param, save_fsdp_optim_param
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload


# Toy for moe
class DummyExperts(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.w2(self.w1(x))


class DummyBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bw1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.bw2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.experts = DummyExperts(hidden_size)

    def forward(self, x):
        y = self.bw2(self.bw1(x))
        return self.experts(y)


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mw1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.mw2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.layers = torch.nn.ModuleList(DummyBlock(hidden_size) for _ in range(3))

        inv_freq = torch.ones((2, 2))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        x = self.mw2(self.mw1(x))
        for layer in self.layers:
            x = layer(x)
        return x


def get_model(hidden_size, seed=123, meta=False, offload_disk=False):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if offload_disk:
        with init_empty_weights_with_disk_offload():
            model = DummyModel(hidden_size)
    else:
        if meta:
            with torch.device("meta"):
                model = DummyModel(hidden_size)
        else:
            model = DummyModel(hidden_size).cuda()
    return model


def get_input(hidden_size, seed=123):
    diff_seed = seed + torch.distributed.get_rank()
    torch.cuda.manual_seed(diff_seed)
    torch.manual_seed(diff_seed)
    return torch.randn(4, hidden_size, device=torch.device("cuda"))


def optim_param_func(model):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_optim(model):
    def optim_param_func(model):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optim = torch.optim.AdamW(optim_param_func(model), lr=0.001)
    return optim


def assert_same_sd(ref_sd, load_sd, strict=True):
    assert set(ref_sd.keys()) == set(load_sd.keys()), (
        f"{[k for k in ref_sd.keys() if k not in load_sd.keys()]} "
        f"{[k for k in load_sd.keys() if k not in ref_sd.keys()]}"
    )
    for k in ref_sd.keys():
        if isinstance(ref_sd[k], dict):
            assert_same_sd(ref_sd[k], load_sd[k], strict)
        elif isinstance(ref_sd[k], torch.Tensor):
            if strict:
                assert torch.all(ref_sd[k] == load_sd[k]), f"{k}\nref_sd\n{ref_sd[k]}\nload_sd\n{load_sd[k]}"
            else:
                assert torch.allclose(
                    ref_sd[k], load_sd[k], rtol=1e-02
                ), f"{k}\nref_sd\n{ref_sd[k]}\nload_sd\n{load_sd[k]}"


cls_to_fsdp_group_config = {}


def init_moe_group(ep_size=2, ddp1=None, ddp2=None, use_device_mesh=False):
    global cls_to_fsdp_group_config

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()
    # ep_size = divide(world_size, 2)
    if ddp1 is not None and ddp1 > 1:
        fsdp_mode = ([("data", divide(world_size, ddp1)), ("ddp1", ddp1)], None)
    else:
        fsdp_mode = ([("data", torch.distributed.get_world_size())], None)

    create_parallel_group(fsdp_mode, use_device_mesh=use_device_mesh)
    if ddp1 is not None and ddp1 > 1:
        cls_to_fsdp_group_config[DummyBlock] = (parallel_group("data"), parallel_group("ddp1"))
    else:
        cls_to_fsdp_group_config[DummyBlock] = parallel_group("data")

    if ep_size > 1:
        if ddp2 is not None and ddp2 > 1:
            ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size * ddp2)), ("ddp2", ddp2)], None)
        else:
            ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size))], None)

        create_parallel_group(ep_mode, use_device_mesh=use_device_mesh)
        if ddp2 is not None and ddp2 > 1:
            cls_to_fsdp_group_config[DummyExperts] = (parallel_group("expert_fsdp"), parallel_group("ddp2"))
        else:
            cls_to_fsdp_group_config[DummyExperts] = parallel_group("expert_fsdp")
    else:
        expert_fsdp_mode = ([("expert_fsdp", world_size)], None)
        create_parallel_group(expert_fsdp_mode, use_device_mesh=use_device_mesh)
        cls_to_fsdp_group_config[DummyExperts] = parallel_group("expert_fsdp")

    return cls_to_fsdp_group_config


def moe_hsdp_policy_fn(module):
    cls_to_fsdp_group = {DummyBlock: parallel_group("data"), DummyExperts: parallel_group("expert_fsdp")}
    # cls_to_fsdp_group = {DummyBlock: parallel_group("data")}

    if module.__class__ in cls_to_fsdp_group:
        pg = cls_to_fsdp_group[module.__class__]
        if isinstance(pg, tuple):
            return {"process_group": pg, "sharding_strategy": ShardingStrategy.HYBRID_SHARD}
        return {"process_group": pg}
    return False


def _init_group_and_model(rank, world_size, hidden_size, ddp1_size=None, ddp2_size=None, ep_size=2):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(ep_size=ep_size, ddp1=ddp1_size, ddp2=ddp2_size)
    m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn),
    }
    if ddp1_size is not None and ddp1_size > 1:
        fsdp_config.update(
            {
                "process_group": (parallel_group("data"), parallel_group("ddp1")),
                "sharding_strategy": ShardingStrategy.HYBRID_SHARD,
            }
        )

    strategy = [("fsdp", fsdp_config)]

    status, result, best_strategy = auto_accelerate(
        m,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model = result.model
    optim = result.optim

    loss = model(get_input(hidden_size)).mean()
    loss.backward()
    optim.step()

    return model, optim, fsdp_config


def run_module_gt(rank, world_size, hidden_size, path, ddp1_size=None, ddp2_size=None, async_save=False, ep_size=2):
    model, optim, _ = _init_group_and_model(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        ddp1_size=ddp1_size,
        ddp2_size=ddp2_size,
        ep_size=ep_size,
    )

    _, ranks = parallel_group_and_ranks("ddp1")
    if ranks is None or rank == ranks[0]:
        if async_save:
            pass
            # pg = parallel_group("data")
            # save_checkpoint(step=1, model=model, optimizer=optim, path=path, group=pg)
        else:
            save_fsdp_flat_param(model, path)
            save_fsdp_optim_param(model, optim, path)
