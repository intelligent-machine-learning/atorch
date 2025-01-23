import argparse
import functools
import inspect
import time
from contextlib import nullcontext
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleInterleaved1F1B
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaRMSNorm

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.common.util_func import data_float_to_dtype, data_to_device
from atorch.distributed.distributed import create_parallel_group, destroy_parallel_group
from atorch.pipeline_parallel.pipe_module import PipeModuleConfig, make_pipe_module
from atorch.pipeline_parallel.pipe_partition import get_rank_stage_info
from atorch.pipeline_parallel.pipe_schedule import make_pipe_schedule_from_pipe_module
from atorch.utils.gc import ManualGarbageCollection
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload
from atorch.utils.version import torch_version

if torch_version() >= (2, 5, 0):  # type: ignore
    from torch.distributed.pipelining import ScheduleInterleavedZeroBubble
else:
    ScheduleInterleavedZeroBubble = None  # type: ignore


class LlamaChunk(LlamaModel):
    def __init__(self, config, layer_num=None, pre_process=True, post_process=True):
        nn.Module.__init__(self)
        self.config = config

        requires_layer_idx = "layer_idx" in inspect.signature(LlamaDecoderLayer.__init__).parameters

        if pre_process:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        else:
            self.embed_tokens = None

        if layer_num is None:
            self.layer_num = config.num_hidden_layers
        else:
            self.layer_num = layer_num

        if requires_layer_idx:
            self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx=None) for _ in range(self.layer_num)])
        else:
            self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(self.layer_num)])

        if post_process:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

    def forward(self, input_ids):
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_ids

        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            hidden_states = hidden_states.to(target_dtype)

        position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids)[0]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states


def get_llama_model_chunk(
    model_config, layer_num=None, pre_process=True, post_process=True, meta_init=False, start_layer_idx=None
):
    context = nullcontext()
    if meta_init:
        context = init_empty_weights_with_disk_offload(ignore_tie_weights=False)
    with context:
        return LlamaChunk(config=model_config, layer_num=layer_num, pre_process=pre_process, post_process=post_process)


def llama_loss_func(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def prepare_input(data, device):
    return data_to_device(data, device)


def get_stage_modules(model_config, stage_ids, pp_stage_num, stage_layer_nums, meta_init=False):
    stages = []
    for stage_id, layer_num in zip(stage_ids, stage_layer_nums):
        pre_process = stage_id == 0
        post_process = stage_id == pp_stage_num - 1
        stage = get_llama_model_chunk(model_config, layer_num, pre_process, post_process, meta_init)
        stages.append(stage)
    return stages


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    if len(samples[0]) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
        null_dataset=False,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.null_dataset = null_dataset

    def __getitem__(self, index):
        if self.null_dataset:
            return {}
        length = self.max_source_positions
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


def get_dataset(seq_length=128, vocab_size=32000, datasize=1000, null_dataset=False):
    return BenchmarkLMDataset(
        vocab_size=vocab_size, max_source_positions=seq_length, total_samples=datasize, null_dataset=null_dataset
    )


def get_dataloader(dataset, batch_size, rank, dp_size, use_distributed):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": False, "num_workers": 2}
    input_name = "input_ids"
    label_name = "labels"
    dataloader_args["collate_fn"] = functools.partial(
        collate_sentences_lm, input_name=input_name, label_name=label_name
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    return dataloader


def optim_grouped_param_func(model):
    no_decay = "bias"
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if no_decay not in n],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if no_decay in n],
            "weight_decay": 0.0,
        },
    ]
    return parameters


class OptimizersContainer:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def state_dict(self):
        state_dict = {}
        for i, optim in self.optimizers:
            state_dict[i] = optim.state_dict()

    def load_state_dict(self, state_dict):
        for i, optim in self.optimizers:
            optim.load_state_dict(state_dict[i])


def pre_create_group(pp_size):
    world_size = torch.distributed.get_world_size()
    dp_size = world_size // pp_size
    if dp_size > 1:
        config = ([("data", dp_size), ("pipe", pp_size)], None)
    else:
        config = ([("pipe", pp_size)], None)
    create_parallel_group(config)


def apply_strategy(stage_ids, stage_num, modules, loss_fn, args):
    res_modules = []
    res_optims = []
    res_loss_fn = None
    optim_func = torch.optim.AdamW
    optim_args = {"lr": 0.001}
    optim_param_func = optim_grouped_param_func if args.optim_grouped_params else None

    strategy = get_strategy(args)

    st = time.time()

    for idx, module in enumerate(modules):
        if stage_ids[idx] == stage_num - 1:
            m_loss_fn = loss_fn
        else:
            m_loss_fn = None
        status, result, _ = auto_accelerate(
            module,
            optim_func=optim_func,
            dataset=None,
            loss_func=m_loss_fn,
            optim_args=optim_args,
            optim_param_func=optim_param_func,
            dataloader_args=None,
            load_strategy=strategy,
        )
        assert status
        res_modules.append(result.model)
        res_optims.append(result.optim)
        if result.loss_func is not None:
            res_loss_fn = result.loss_func

    if atorch.rank() == 0:
        print("auto_accelerate time : ", time.time() - st)

    return res_modules, OptimizersContainer(res_optims), res_loss_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--hidden_size", type=int, default=256, required=False)
    parser.add_argument("--intermediate_size", type=int, default=512, required=False)
    parser.add_argument("--head_num", type=int, default=4, required=False)
    parser.add_argument("--key_value_head_num", type=int, default=4, required=False)
    parser.add_argument("--layer_num", type=int, default=16, required=False)
    parser.add_argument("--seq_length", type=int, default=64, required=False)

    parser.add_argument("--batchsize_per_gpu", type=int, default=8, required=False)
    parser.add_argument("--microbatchsize", type=int, default=1, required=False)
    parser.add_argument("--max_train_step", type=int, default=20, required=False)

    parser.add_argument("--optim_grouped_params", default=False, action="store_true")
    parser.add_argument("--log_interval", type=int, default=10, required=False)
    parser.add_argument("--timeout_sec", type=int, default=1800, required=False)

    parser.add_argument("--pp_num", type=int, default=4, required=False)
    parser.add_argument("--pp_stage_num", type=int, default=-1, required=False)
    parser.add_argument(
        "--pp_schedule",
        type=str,
        default="Interleaved1F1B",
        required=False,
        help="supported schedule: 1F1B, Interleaved1F1B, etc",
    )
    parser.add_argument(
        "--pp_stage_layer_num",
        type=str,
        default="",
        required=False,
        help="A semicolon-delimited string of integer key-value pairs, where keys and values are separated by a colon;"
        "key is the stage_id and value is the corresponding layer num. These stages would use corresponding layer nums,"
        "and other stages would evenly partition the remaining layers."
        "This can be used for uneven pp partition."
        "For example, '0:1;3:2' means stage 0 has 1 layer and stage 3 has 2 layers. "
        "If total layers are 9 and total stages are 4, stage 1 and 2 would have 3 layers.",
    )
    parser.add_argument("--use_fsdp", default=False, action="store_true")
    parser.add_argument("--use_fp8", default=False, action="store_true")
    parser.add_argument("--use_checkpointing", default=False, action="store_true")
    parser.add_argument("--use_module_replace", default=False, action="store_true")
    parser.add_argument("--use_meta_init", default=False, action="store_true")
    parser.add_argument("--use_distributed_dataloader", default=False, action="store_true")
    parser.add_argument("--max_checkpoint_module_num", type=int, default=-1, required=False)
    parser.add_argument("--use_atorch_pp", default=False, action="store_true")
    parser.add_argument("--npu", default=False, action="store_true")
    parser.add_argument("--record_timeline", default=False, action="store_true")
    parser.add_argument("--timeline_dir", type=str, default="timeline_dir", required=False)
    parser.add_argument("--record_memory_history", default=False, action="store_true")

    return parser.parse_args()


def get_strategy(args):
    strategy = []
    if args.use_module_replace:
        strategy.append("module_replace")
    if args.use_fsdp:
        print("fsdp to be supporeded, ignored now.")
    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy.append(("amp_native", amp_config))
    if args.use_checkpointing:
        checkpoint_modules = (LlamaDecoderLayer,)
        checkpoint_config = {"wrap_class": checkpoint_modules, "no_reentrant": True}
        if args.max_checkpoint_module_num >= 0:
            checkpoint_config["max_checkpoint_module_num"] = args.max_checkpoint_module_num
        strategy.append(("checkpoint", checkpoint_config))
    if args.use_fp8:
        strategy.append(("fp8", {"include": ("layers",)}))
    return strategy


def print_model_size(rank, stage_ids, stage_modules):
    total_params = 0
    for s_id, module in zip(stage_ids, stage_modules):
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        print(f"[rank {rank}] (stage_id = {s_id}) number of parameters : {params}")
    print(f"[rank {rank}] Total params: {total_params}")


def train_model(
    dataloader,
    optim,
    schedule,
    args,
    global_step,
    start_time,
    device,
    is_first_stage,
    is_last_stage,
    total_batchsize,
    gc_handler,
    step_time_stats,
    throughput_stats,
    max_reserved_stats,
    max_allocated_stats,
    prof=None,
):
    for batch in dataloader:
        gc_handler.run(global_step)
        optim.zero_grad()
        batch = prepare_input(batch, device)
        if is_first_stage:
            schedule.step(batch["input_ids"])
        elif is_last_stage:
            schedule.step(target=batch["labels"])
        else:
            schedule.step()
        optim.step()
        if prof is not None:
            prof.step()
        global_step += 1
        if global_step % args.log_interval == 0 and atorch.rank() == 0:
            cur_time = time.time()
            time_per_step = (cur_time - start_time) / args.log_interval
            sample_per_second = total_batchsize / time_per_step
            print(f"[step={global_step-1}]: {time_per_step} sec/step, throughput is {sample_per_second} sample/sec.")
            mem_reserved = torch.cuda.max_memory_reserved() / 1e6
            mem_allcoated = torch.cuda.max_memory_allocated() / 1e6
            print(f"max_memory_reserved={mem_reserved} MB, max_memory_allocated={mem_allcoated} MB.")
            torch.cuda.reset_peak_memory_stats()
            step_time_stats.append(time_per_step)
            throughput_stats.append(sample_per_second)
            max_reserved_stats.append(mem_reserved)
            max_allocated_stats.append(mem_allcoated)
            start_time = cur_time


def get_manual_stage_partition(args):
    manual_stage_partition = None
    if args.pp_stage_layer_num != "":
        manual_stage_partition = {}
        pairs = args.pp_stage_layer_num.split(";")
        for pair in pairs:
            stage_idx, layer_num = pair.split(":")
            manual_stage_partition[int(stage_idx)] = int(layer_num)
    return manual_stage_partition


def train(args):
    timeout = timedelta(seconds=args.timeout_sec)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True, timeout=timeout)
    if atorch.rank() == 0:
        print("Args is ", args)

    # create data + pipe pg
    pre_create_group(args.pp_num)

    rank = atorch.rank()
    pp_rank = atorch.distributed.distributed.parallel_rank("pipe")
    pp_size = atorch.distributed.distributed.parallel_group_size("pipe")
    dp_rank = atorch.distributed.distributed.parallel_rank("data")
    dp_size = atorch.distributed.distributed.parallel_group_size("data") if dp_rank is not None else 1
    is_first_stage = pp_rank == 0
    is_last_stage = pp_rank == pp_size - 1
    pp_stage_num = args.pp_num if args.pp_stage_num <= 0 else args.pp_stage_num

    model_config = LlamaConfig()
    c_s = f"hidden_size={args.hidden_size},num_attention_heads={args.head_num},num_hidden_layers={args.layer_num},"
    c_s += f"num_key_value_heads={args.key_value_head_num},max_position_embeddings={args.seq_length},"
    c_s += f"intermediate_size={args.intermediate_size}"
    model_config.update_from_string(c_s)
    model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = False

    batchsize_per_gpu = args.batchsize_per_gpu
    total_batchsize = batchsize_per_gpu * dp_size
    total_data_size = args.max_train_step * total_batchsize
    microbatch_num = batchsize_per_gpu // args.microbatchsize

    if args.use_distributed_dataloader:
        datasize = total_data_size
    else:
        datasize = total_data_size // dp_size

    # create dataset/dataloader
    def create_dataloader(stage_ids):
        null_dataset = 0 not in stage_ids and pp_stage_num - 1 not in stage_ids
        dataset = get_dataset(
            seq_length=args.seq_length, vocab_size=model_config.vocab_size, datasize=datasize, null_dataset=null_dataset
        )

        dataloader = get_dataloader(dataset, batchsize_per_gpu, dp_rank, dp_size, args.use_distributed_dataloader)
        return dataloader, null_dataset

    # create model
    st = time.time()

    if atorch.rank() == 0:
        print(model_config)

    loss_fn = functools.partial(llama_loss_func, vocab_size=model_config.vocab_size)

    device = atorch.local_rank()
    mp_precision = torch.bfloat16
    stage_group = atorch.distributed.distributed.parallel_group("pipe")

    manual_stage_partition = get_manual_stage_partition(args)

    if args.use_atorch_pp:
        activation_mapping = [("hidden_states", 0)]
        batch_mapping = [("input_ids", 0)]
        default_input_info = (activation_mapping, None)
        stage_0_input_info = (None, batch_mapping)
        io_mapping = {"default": default_input_info, 0: stage_0_input_info}

        # create PipeModuleConfig
        pm_config = PipeModuleConfig(
            model_config=model_config,
            manual_stage_partition=manual_stage_partition,
            total_layer_num=args.layer_num,
            tie_weight_info=None,
            input_output_mapping=io_mapping,
            virtual_pp_size=pp_stage_num // args.pp_num,
            auto_ddp=True,
            sche_name="Schedule" + args.pp_schedule,
            n_microbatches=microbatch_num,
        )

        # create pipe_module
        accelerate_strategy = get_strategy(args)
        pipe_module = make_pipe_module(
            model_provider=get_llama_model_chunk, loss_func=loss_fn, strategy=accelerate_strategy, config=pm_config
        )

        stage_ids = pipe_module.stage_ids

        # create schedule
        schedule = make_pipe_schedule_from_pipe_module(pipe_module, pm_config)

        # create optimizer
        optim_func = torch.optim.AdamW
        optim_args = {"lr": 0.001}
        if args.optim_grouped_params:
            optim = optim_func(optim_grouped_param_func(pipe_module), **optim_args)
        else:
            optim = optim_func(pipe_module.parameters(), **optim_args)

        dataloader, _ = create_dataloader(stage_ids)
    else:
        # total_layer_num, pp_rank, pp_size, virtual_pp_size, style="loop", manual_stage_partition=None)
        stage_info = get_rank_stage_info(
            args.layer_num,
            pp_rank,
            args.pp_num,
            pp_stage_num // args.pp_num,
            manual_stage_partition=manual_stage_partition,
        )

        stage_ids = [idx for (idx, _, _) in stage_info]
        stage_layer_nums = [layer_num for (_, layer_num, _) in stage_info]
        # get pp stage modules for current rank
        stage_modules = get_stage_modules(
            model_config, stage_ids, pp_stage_num, stage_layer_nums, meta_init=args.use_meta_init
        )

        if dp_rank == 0:
            print_model_size(rank, stage_ids, stage_modules)

        # apply strategy to modules
        stage_modules, optim, loss_func = apply_strategy(stage_ids, pp_stage_num, stage_modules, loss_fn, args)

        dataloader, null_dataset = create_dataloader(stage_ids)

        split_batch = None
        if not null_dataset:
            # get one example data batch
            data_iter = iter(dataloader)
            one_batch = next(data_iter)
            _, split_batch = split_args_kwargs_into_chunks((), one_batch, microbatch_num)
            split_batch = split_batch[0]

        block_io_shape = (args.microbatchsize, args.seq_length, args.hidden_size)
        block_io_tensor = data_to_device(torch.empty(block_io_shape), device)
        output_shape = (args.microbatchsize, args.seq_length, model_config.vocab_size)
        output_tensor = data_to_device(torch.empty(output_shape), device)

        # create stages
        stages = []
        for s_id, module in zip(stage_ids, stage_modules):
            if s_id == 0:
                input_args = data_to_device(split_batch["input_ids"], device)
            else:
                input_args = data_float_to_dtype(block_io_tensor, mp_precision)
            if s_id == pp_stage_num - 1:
                output_args = data_float_to_dtype(output_tensor, mp_precision)
            else:
                output_args = data_float_to_dtype(block_io_tensor, mp_precision)

            stage = PipelineStage(
                module,
                s_id,
                pp_stage_num,
                device,
                input_args=input_args,
                output_args=output_args,
                group=stage_group,
            )
            stages.append(stage)

        del block_io_tensor
        del output_tensor

        # create pp schedule
        if args.pp_schedule == "1F1B":
            schedule_cls = Schedule1F1B
            assert pp_size == pp_stage_num, "1F1B pp stage num should be same as pp size"
            assert len(stages) == 1
            stages = stages[0]
        elif args.pp_schedule == "Interleaved1F1B":
            schedule_cls = ScheduleInterleaved1F1B
        elif args.pp_schedule == "ZeroBubble":
            schedule_cls = ScheduleInterleavedZeroBubble

        # Last stage has loss_func, and other stages use original loss_fn to indicate that pp requires backward.
        schedule = schedule_cls(
            stages, n_microbatches=microbatch_num, loss_fn=loss_func if loss_func is not None else loss_fn
        )

    if atorch.rank() == 0:
        print("Get model time : ", time.time() - st)

    global_step = 0
    gc_handler = ManualGarbageCollection(gc_freq=10, disable_auto_gc=True)
    start_time = time.time()

    step_time_stats = []
    throughput_stats = []
    max_reserved_stats = []
    max_allocated_stats = []

    if args.record_timeline:
        with torch.profiler.profile(
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,  # Skip first N steps, profiler is disabled
                warmup=1,  # Warmup steps, profiler is enabled, but results are discarded
                active=5,  # Profiler works and records events.
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.timeline_dir),
        ) as prof:
            train_model(
                dataloader,
                optim,
                schedule,
                args,
                global_step,
                start_time,
                device,
                is_first_stage,
                is_last_stage,
                total_batchsize,
                gc_handler,
                step_time_stats,
                throughput_stats,
                max_reserved_stats,
                max_allocated_stats,
                prof,
            )
    elif args.record_memory_history:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"visual_mem_{timestamp}.pickle"
        try:
            train_model(
                dataloader,
                optim,
                schedule,
                args,
                global_step,
                start_time,
                device,
                is_first_stage,
                is_last_stage,
                total_batchsize,
                gc_handler,
                step_time_stats,
                throughput_stats,
                max_reserved_stats,
                max_allocated_stats,
            )
        except Exception as e:
            print(str(e))
        finally:
            torch.cuda.memory._dump_snapshot(file_name)
            torch.cuda.memory._record_memory_history(enabled=None)
    else:
        train_model(
            dataloader,
            optim,
            schedule,
            args,
            global_step,
            start_time,
            device,
            is_first_stage,
            is_last_stage,
            total_batchsize,
            gc_handler,
            step_time_stats,
            throughput_stats,
            max_reserved_stats,
            max_allocated_stats,
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
    print(args)
    if args.npu:
        from atorch import npu  # noqa
    train(args)
