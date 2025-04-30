import argparse
import functools
import time
from datetime import timedelta

import torch
from moe_modules import (
    get_dataloader,
    get_dataset,
    get_model,
    llama_loss_func,
    moe_fsdp2_policy_fn,
    optim_grouped_param_func,
    patch_llama,
    prepare_input,
    set_global_variable_from_args,
)
from torch.distributed.fsdp.wrap import CustomPolicy
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.model_context import get_data_partition_rank_and_size
from atorch.checkpoint.torch_checkpoint import TrainState
from atorch.common.util_func import data_to_device
from atorch.distributed.distributed import destroy_parallel_group
from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.moe_util import (
    _set_moe_forward_prefetch_for_fsdp2_ep,
    clip_grad_norm_,
    compute_grad_norm_,
    compute_param_norm_,
    set_inter_fsdp_state,
)
from atorch.utils.version import torch_version


def pre_create_group(ep_size, use_device_mesh):
    from atorch.common.util_func import divide
    from atorch.distributed.distributed import create_parallel_group

    world_size = torch.distributed.get_world_size()
    fsdp_mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(fsdp_mode, use_device_mesh=use_device_mesh)
    ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size))], None)
    create_parallel_group(ep_mode, use_device_mesh=use_device_mesh)


def parse_args():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--ep_size", type=int, default=1, required=False)
    parser.add_argument("--num_experts", type=int, default=8, required=False)
    parser.add_argument("--top_k", type=int, default=2, required=False)
    parser.add_argument("--num_shared_expert", type=int, default=2, required=False)
    parser.add_argument("--hidden_size", type=int, default=256, required=False)
    parser.add_argument("--intermediate_size", type=int, default=512, required=False)
    parser.add_argument("--head_num", type=int, default=4, required=False)
    parser.add_argument("--key_value_head_num", type=int, default=4, required=False)
    parser.add_argument("--layer_num", type=int, default=3, required=False)
    parser.add_argument("--seq_length", type=int, default=64, required=False)

    parser.add_argument("--batchsize_per_gpu", type=int, default=8, required=False)
    parser.add_argument("--max_train_step", type=int, default=20, required=False)

    parser.add_argument("--optim_grouped_params", default=False, action="store_true")
    parser.add_argument("--log_interval", type=int, default=10, required=False)
    parser.add_argument("--timeout_sec", type=int, default=1800, required=False)

    parser.add_argument("--use_fsdp", default=False, action="store_true")
    parser.add_argument("--use_fsdp2", default=False, action="store_true")
    parser.add_argument("--use_amp", default=False, action="store_true")
    parser.add_argument("--use_fp8", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument("--use_module_replace", default=False, action="store_true")
    parser.add_argument("--not_use_atorch_rmsnorm", default=False, action="store_true")
    parser.add_argument("--use_meta_init", default=False, action="store_true")
    parser.add_argument("--use_distributed_dataloader", default=False, action="store_true")
    parser.add_argument("--shared_expert_overlapping", default=False, action="store_true")
    parser.add_argument("--max_checkpoint_module_num", type=int, default=-1, required=False)
    parser.add_argument("--record_timeline", default=False, action="store_true")
    parser.add_argument("--timeline_dir", type=str, default="timeline_dir", required=False)
    parser.add_argument(
        "--moe_implementation_type",
        type=str,
        default="Megatron",
        required=False,
        help="supported value: Megatron, MegaBlocks",
    )
    parser.add_argument(
        "--moe_token_dispatcher_type",
        type=str,
        default="MindSpeedAllGather",
        required=False,
        help="supported value: AllToAll, AllGather, MindSpeedAllToAll, MindSpeedAllGather",
    )
    parser.add_argument("--npu", default=False, action="store_true")
    parser.add_argument("--no_reentrant", default=True, action="store_true")
    parser.add_argument("--clip_grad", default=False, action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--open_fsdp2_prefetch", default=False, action="store_true")

    return parser.parse_args()


def get_strategy(args):
    if args.ep_size > 1:
        strategy = []
    else:
        parallel_mode = ([("data", torch.distributed.get_world_size())], None, False, True)
        strategy = [("parallel_mode", parallel_mode)]

    if args.use_module_replace:
        strategy.append("module_replace")

    atorch_wrap_cls = (LlamaDecoderLayer,)
    if args.use_fsdp:
        fsdp_config = {
            "sync_module_states": True,
            "limit_all_gathers": True,
            "forward_prefetch": True,
        }

        if args.ep_size > 1:
            from atorch.distributed.distributed import parallel_group

            experts_cls = Grouped_GEMM_MoE

            def moe_fsdp_policy_fn(module):
                if isinstance(module, atorch_wrap_cls):
                    # non experts fsdp wrap
                    return {"process_group": parallel_group("data")}
                elif isinstance(module, experts_cls):
                    # experts fsdp wrap
                    return {"process_group": parallel_group("expert_fsdp")}
                return False

            moe_fsdp_policy = CustomPolicy(moe_fsdp_policy_fn)
            fsdp_config["auto_wrap_policy"] = moe_fsdp_policy
        else:
            fsdp_config["atorch_wrap_cls"] = atorch_wrap_cls
        strategy.append(("fsdp", fsdp_config))
        if args.optim_grouped_params:
            fsdp_config["use_orig_params"] = True
    elif args.use_fsdp2:
        if args.ep_size > 1:
            nested_fsdp2_policy = functools.partial(moe_fsdp2_policy_fn, nested_fsdp2=True)
            fsdp_config = {
                "auto_wrap_policy": CustomPolicy(nested_fsdp2_policy),
            }
        else:
            fsdp_config = {
                "atorch_wrap_cls": atorch_wrap_cls,
            }

        strategy.append(("fsdp2", fsdp_config))

    if args.use_amp:
        amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
        strategy.append(("amp_native", amp_config))
    if args.use_checkpointing:
        checkpoint_modules = (LlamaDecoderLayer,)
        no_reentrant = args.no_reentrant
        # Note: no_reentrant must be True when ep + fsdp2&prefetch.
        if args.use_fsdp2 and args.ep_size > 1 and not no_reentrant and args.open_fsdp2_prefetch:
            print("no_reentrant must be True when ep + fsdp2&prefetch, change no_reentrant to True.")
            no_reentrant = True
        elif args.use_fsdp2 and args.ep_size > 1 and no_reentrant and not args.open_fsdp2_prefetch:
            print("no_reentrant must be False when ep + fsdp2&no prefetch, change no_reentrant to False.")
            no_reentrant = False
        checkpoint_config = {"wrap_class": checkpoint_modules, "no_reentrant": no_reentrant}
        if args.max_checkpoint_module_num >= 0:
            checkpoint_config["max_checkpoint_module_num"] = args.max_checkpoint_module_num
        strategy.append(("checkpoint", checkpoint_config))
    if args.use_fp8:
        strategy.append(("fp8", {"include": ("layers",)}))
    return strategy


def train_model(
    args,
    dataloader,
    optim,
    model,
    loss_func,
    total_batchsize,
    step_time_stats,
    throughput_stats,
    max_reserved_stats,
    max_allocated_stats,
    train_state,
    prof=None,
):
    start_time = time.time()
    device = "cuda"

    for batch in dataloader:
        optim.zero_grad()
        batch = prepare_input(batch, device)
        outputs = model(**batch)
        loss = loss_func(batch, outputs)
        loss.backward()

        if args.use_fsdp2 and args.clip_grad:
            clip_grad_norm_(model.parameters(), args.max_grad_norm, foreach=True)
            grad_norm = compute_grad_norm_(model.parameters(), foreach=True)
            param_norm = compute_param_norm_(model.parameters(), foreach=True)

        optim.step()
        if prof is not None:
            prof.step()
        train_state.step += 1
        if train_state.step % args.log_interval == 0 and atorch.rank() == 0:
            cur_time = time.time()
            time_per_step = (cur_time - start_time) / args.log_interval
            sample_per_second = total_batchsize / time_per_step
            print(
                f"[step={train_state.step-1}]: {time_per_step} sec/step, throughput is {sample_per_second} sample/sec."
            )
            if args.use_fsdp2 and args.clip_grad:
                print(f"grad_norm={grad_norm.item()}, param_norm={param_norm.item()}")
            mem_reserved = torch.cuda.max_memory_reserved() / 1e6
            mem_allcoated = torch.cuda.max_memory_allocated() / 1e6
            print(f"max_memory_reserved={mem_reserved} MB, max_memory_allocated={mem_allcoated} MB.")
            torch.cuda.reset_peak_memory_stats()
            step_time_stats.append(time_per_step)
            throughput_stats.append(sample_per_second)
            max_reserved_stats.append(mem_reserved)
            max_allocated_stats.append(mem_allcoated)
            start_time = cur_time


def train(args):
    timeout = timedelta(seconds=args.timeout_sec)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True, timeout=timeout)
    if atorch.rank() == 0:
        print("Args is ", args)

    model_config = LlamaConfig()
    c_s = f"hidden_size={args.hidden_size},num_attention_heads={args.head_num},num_hidden_layers={args.layer_num},"
    c_s += f"num_key_value_heads={args.key_value_head_num},max_position_embeddings={args.seq_length},"
    c_s += f"intermediate_size={args.intermediate_size}"
    model_config.update_from_string(c_s)
    if is_torch_npu_available() and torch_version() >= (2, 4, 0):
        model_config._attn_implementation = "sdpa"
    else:
        model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = False
    model_config.num_experts = args.num_experts
    model_config.num_shared_expert = args.num_shared_expert
    model_config.top_k = args.top_k
    model_config.shared_expert_overlapping = args.shared_expert_overlapping
    model_config.ep_size = args.ep_size
    model_config.moe_aux_loss_coeff = 0.01
    model_config.moe_z_loss_coeff = 0.0002

    st = time.time()
    if args.ep_size > 1:
        pre_create_group(args.ep_size, use_device_mesh=args.use_fsdp2)
    model = get_model(model_config, meta_init=args.use_meta_init)
    if atorch.rank() == 0:
        print("Get model time : ", time.time() - st)

    if atorch.rank() == 0:
        print(model_config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

    optim_func = torch.optim.AdamW
    optim_args = {"lr": 0.001}
    optim_param_func = optim_grouped_param_func if args.optim_grouped_params else None

    prepare_input = data_to_device

    strategy = get_strategy(args)

    st = time.time()
    # auto_accelerate
    status, res, _ = auto_accelerate(
        model,
        optim_func=optim_func,
        dataset=None,
        loss_func=llama_loss_func,
        prepare_input=prepare_input,
        model_input_format="unpack_dict",
        optim_args=optim_args,
        optim_param_func=optim_param_func,
        dataloader_args=None,
        load_strategy=strategy,
    )
    assert status
    if atorch.rank() == 0:
        print("auto_accelerate time : ", time.time() - st)

    model = res.model
    optim = res.optim
    dataloader = res.dataloader
    loss_func = res.loss_func
    prepare_input = res.prepare_input

    if args.ep_size > 1 and args.use_fsdp2:
        set_inter_fsdp_state(model, LlamaDecoderLayer, Grouped_GEMM_MoE)
        _set_moe_forward_prefetch_for_fsdp2_ep(model, LlamaDecoderLayer, Grouped_GEMM_MoE, LlamaForCausalLM)

    train_state = TrainState()

    rank, dp_size = get_data_partition_rank_and_size()
    batchsize_per_gpu = args.batchsize_per_gpu
    total_batchsize = batchsize_per_gpu * dp_size
    total_data_size = args.max_train_step * total_batchsize

    if args.use_distributed_dataloader:
        datasize = total_data_size
    else:
        datasize = total_data_size // dp_size
    dataset = get_dataset(seq_length=args.seq_length, vocab_size=model_config.vocab_size, datasize=datasize)
    dataloader = get_dataloader(dataset, batchsize_per_gpu, rank, dp_size, args.use_distributed_dataloader)

    step_time_stats = []
    throughput_stats = []
    max_reserved_stats = []
    max_allocated_stats = []

    if args.record_timeline:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=5,  # Skip first N steps, profiler is disabled
                warmup=5,  # Warmup steps, profiler is enabled, but results are discarded
                active=5,  # Profiler works and records events.
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.timeline_dir),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            train_model(
                args,
                dataloader,
                optim,
                model,
                loss_func,
                total_batchsize,
                step_time_stats,
                throughput_stats,
                max_reserved_stats,
                max_allocated_stats,
                train_state,
                prof=prof,
            )
    else:
        train_model(
            args,
            dataloader,
            optim,
            model,
            loss_func,
            total_batchsize,
            step_time_stats,
            throughput_stats,
            max_reserved_stats,
            max_allocated_stats,
            train_state,
        )

    if atorch.rank() == 0:
        print("Finished training!")
        total_valid_stats_num = len(step_time_stats) - 1
        if total_valid_stats_num > 0:
            avg_step_time = sum(step_time_stats[1:]) / total_valid_stats_num
            avg_throughput = sum(throughput_stats[1:]) / total_valid_stats_num
            avg_throughput_token = batchsize_per_gpu * args.seq_length / avg_step_time
            avg_max_reserved = max(max_reserved_stats)
            avg_max_allocated = max(max_allocated_stats)
            print(f"Average : {avg_step_time} sec/step.")
            print(f"Average thoughput: {avg_throughput} sample/sec, {avg_throughput_token} token/gpu/sec.")
            print(f"max_memory_reserved={avg_max_reserved} MB, max_memory_allocated={avg_max_allocated} MB.")
    torch.distributed.barrier()
    destroy_parallel_group()


if __name__ == "__main__":
    args = parse_args()
    set_global_variable_from_args(args)
    if args.npu:
        from atorch import npu  # noqa
    patch_llama(args)
    train(args)
