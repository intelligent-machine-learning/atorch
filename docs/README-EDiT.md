<h1 align="center"><b>EDiT</b></h1>
<h3 align="center"><b>EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models</b></h3>

[![ICLR](https://img.shields.io/badge/ICLR-2025-blue)](https://arxiv.org/abs/2412.07210)
[![arXiv](https://img.shields.io/badge/arXiv-2412.07210-b31b1b.svg)](https://arxiv.org/abs/2412.07210)

We present PyTorch code for [EDiT: A Local-SGD-Based Efficient Distributed Training Method for Large Language Models](https://arxiv.org/abs/2412.07210), ICLR'25.

# Introduction


Existing distributed training methods often suffer from communication bottlenecks, stragglers, and limited elasticity, particularly in heterogeneous or large-scale environments. Local SGD methods have been proposed to address these issues, but their effectiveness remains limited to small-scale training due to additional memory overhead and concerns on efficiency and stability. 

To tackle these issues, we propose **EDiT**, an innovative **E**fficient **Di**stributed **T**raining method that combines a tailored Local SGD approach with model sharding techniques to
enhance large-scale training efficiency. 
+ EDiT performs layer-wise parameter synchronization during forward pass, reducing communication and memory overhead and enabling the overlap of computation and communication. 
+ EDiT employs a pseudo gradient penalty strategy to suppress loss spikes, which ensures training stability and improve performance. 
+ We introduce **A-EDiT**, a fully asynchronous variant of EDiT that accommodates heterogeneous clusters.

Experimental results demonstrate the superior performance of EDiT/A-EDiT compared with other Local SGD-based methods, establishing them as robust solutions for distributed LLM training in diverse computational ecosystems.

![illustration](./img/edit_illustration.png)

# Usage

## Setup

There are three config classes for initializing the EDiT method, `LocalSGDConfig`, `OuterOptimizerConfig`, and `GTAConfig`. The details are as follows:

## LocalSGDConfig
- `use_async`: Whether use A-EDiT. Default is **False**.
- `local_sgd_sync_interval`: The sync interval step. Default is **1**. (Only works in EDiT)
- `local_sgd_sync_time`: The sync interval time (seconds). Default is **600**. (Only works in A-EDiT)
- `min_total_global_steps`: The minimum total global steps between two synchronization points. The synchronization will be skipped if this number is not reached. Default is **100**. (Only works in A-EDiT)
- `use_step_weight`: Whether to weight the pseudo gradients with the step number. Default is **False**. (Only works in A-EDiT)
- `step_weight_ratio`: The ratio of the step weights towards the pseudo grad norm weight. Default is **0.5**. (Only works in A-EDiT)
- `local_sgd_warmup_steps`: The warmup steps of EDiT. Default is **0**.
- `gradient_accumulation_steps`: The gradient accumulation steps. Default is **1**.
- `clip_pseudo_grad`: The threshold for clipping the pseudo gradients. Usually set **1** or **10**. Default is **None**.
- `pseudo_gradnorm_reduce`: Whether to weight the pseudo gradients based on the pseudo gradient norms. Default is **False**.
- `weight_softmax_temperature`: The temperature in pseudo grad norm weight method. Default is **None**.
- `skip_anomaly`: Whether to skip the abnormal workers when update. Default is **False**.
- `ewma_alpha`: The hyperparameter to control the exponential moving average in anomaly detector. Default is **0.02**.
- `ewma_warmup_steps`: The warmup steps of anomaly detector. Default is **120**.
- `ewma_threshold`: The threshold to find out anomalies in anomaly detector. Default is **3**.
- `cpu_offload`: Whether to offload the outer optimizers and extra parameters to CPU memory. Default is **False**.
- `is_debug`: Whether to print the information about pseudo gradient clipping and anomaly skipping. Default is **False**.

### OuterOptimizerConfig
- `outer_optim_class`: The class of outer optimizer. Default is **None**.
- `outer_optim_kwargs`: The hyperparameters of the outer optimizer. Default is **{}**.

### GTAConfig
- `reducer`: The reducer type with two options: **["linear", "gta"]**. Default is **None**.
- `consensus_method`: The consensus method to unify the directions of the pseudo gradients across different workers with two options: **["sum", "count"]**. Defaults is **None**.
- `sparsification_method`: The sparsification method for sparsifying the pseudo gradients with three options: **["magnitude", "random", "rescaled_random"]**. Default is **None**.
- `normalize`: Whether to normalize the synced pseudo gradients. Default is **True**.
- `density`: The density of the sparse pseudo gradients. Default is **1.0**.
- `int8_mask`: Whether to use INT8 mask when sparsify the pseudo gradients for saving memory. Default is **False**.


## FSDP Integration

### Training
We present the core code for training a model using EDiT in Atorch.

```python
from atorch.local_sgd import GTAConfig, LocalSGDConfig, OuterOptimizerConfig

...

model = YourModel()


# 1. setup the parallel mode
# "zero" corresponds to model shard group, commonly the number of GPUS in a node
# "data" corresponds to model sync group, commonly the number of nodes
parallel_config = ([("zero", model_shard_group_size), ("data", model_sync_group_size)], None)
strategy = [("parallel_mode", parallel_config)]

# 2. setup the fsdp config
fsdp_config = {
    "sync_module_states": True,
    "limit_all_gathers": True,
    "forward_prefetch": True,
    "atorch_wrap_cls": (YourModelLayer,),
    "use_orig_params": True,

}
# 3. insert the EDiT setups
fsdp_config["use_local_sgd"] = True
fsdp_config["local_sgd_config"] = LocalSGDConfig(
    local_sgd_sync_interval=10,
    local_sgd_warmup_steps=100,
    clip_pseudo_grad=10.0,
    pseudo_gradnorm_reduce=True,
    skip_anomaly=True,
    ewma_warmup_steps=10,
)
fsdp_config["outer_optim_config"] = OuterOptimizerConfig(
    outer_optim_class=torch.optim.SGD,
    outer_optim_kwargs={
        "lr": 0.7,
        "momentum": 0.8,
        "nesterov": True,
    },
)
fsdp_config["gta_config"] = GTAConfig(reducer="gta")
strategy.append(("fsdp", fsdp_config))

# 4. auto_accelerate
status, res, _ = auto_accelerate(
    model,
    load_strategy=strategy,
    ...
)
assert status
model = res.model

# 5. train code
...
```

### Checkpoint
We also provide the checkpoint methods for EDiT, which help save and load the state dicts of `global step`, `outer optimizer`, and `anomaly detector`.

```python
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

# 1. save the EDiT checkpoints
FSDP.save_local_sgd_state_dict(
    model,                  # the FSDP-wrapped model
    group=None,             # the model-shard process group (Default: None)
    rank0_only=True,        # whether only gather the state dicts on rank0 (Default: True)
    full_state_dict=False,  # whether gather the full state_dict (Default: False)
    cpu_offload=True,       # whether offload the state dicts to CPU memory (Default: True)
    save_dir="./lsgd_ckpt", # the save dir (Default: "./lsgd_ckpt")
    ckpt_name="lsgd_ckpt",  # the name of saved checkpoints (Default: "lsgd_ckpt")
)

# 2. load the EDiT checkpoints (should be called after the model is wrapped by FSDP!)
FSDP.load_local_sgd_state_dict(
    model,                  # the FSDP-wrapped model
    group=None,             # the model-shard process group 
    rank0_only=True,        # whether only gather the state dicts on rank0 (Default: True)
    full_state_dict=True,  # whether gather the full state_dict (Default: False)
    load_dir="./lsgd_ckpt", # the load dir (Default: "./lsgd_ckpt")
    ckpt_name="lsgd_ckpt",  # the name of loaded checkpoints (Default: "lsgd_ckpt")
)
```

### Example

We provide a small example to illustrate how to use the EDiT method.

```shell
cd examples/local_sgd/auto_accelerate/

# Train llama model with EDiT method.
python -m atorch.distributed.run --nproc_per_node 8 train.py \
--model_type llama \
--datasize 500 \
--distributed \
--hidden_size 64 \
--head_num 4 \
--layer_num 4 \
--seq_length 32 \
--load_strategy \
--use_fsdp \
--use_amp \
--use_module_replace \
--use_local_sgd \
--local_sgd_sync_interval 5 \
--local_sgd_warmup_steps 10 \
--clip_pseudo_grad 10 \
--gradnorm_weighted \
--skip_anomaly \
--skip_anomaly_warmup_steps 10 \
--outer_optim_class sgd
```

## DDP integration
Sharing the same set of config, we also support EDiT style DDP/3D training.

Torch native DDP can be combined with our EDiT optimizer wrapper to enable simple EDiT training.

### Training

```python
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import PostLocalSGDState, post_localSGD_hook
from atorch.local_sgd.DDP import OuterOptimPeriodicModelAverager, StatefulPostLocalSGDOptimizer

...

model = YourModel()


# 1. setup the parallel mode
strategy = ["parallel_mode"]

# 2. auto_accelerate, apply normal DDP wrapper
status, res, _ = auto_accelerate(
    model,
    optim_func=optim_func,
    load_strategy=strategy,
    ...
)
assert status
model = res.model
optim = res.optim

# 3. enable EDiT training with EDiT optim wrapper
local_sgd_state = PostLocalSGDState(
    process_group=parallel_group("data"),
    start_localSGD_iter=args.local_sgd_warmup_steps,
    subgroup=None,
    post_local_gradient_allreduce=False,
)
model.register_comm_hook(local_sgd_state, post_localSGD_hook)
averager = OuterOptimPeriodicModelAverager(
    process_group=parallel_group("data"),
    local_sgd_config=local_sgd_config,
    gta_config=gta_config,
    outer_optim_config=outer_optim_config,
)

optim = StatefulPostLocalSGDOptimizer(
    optim=optim,
    averager=averager,
)
# 5. train code
...
```
EDiT with DDP requires no special handling for EDiT checkpoints, they are taken care of within StatefulPostLocalSGDOptimizer.
Simply
```python
optim_state_dict = optim.state_dict()
```

### Example
```shell
cd examples/local_sgd/auto_accelerate/

# Train llama model with EDiT method. Simply remove --use_fsdp flag
python -m atorch.distributed.run --nproc_per_node 8 train.py \
--model_type llama \
--datasize 500 \
--distributed \
--hidden_size 64 \
--head_num 4 \
--layer_num 4 \
--seq_length 32 \
--load_strategy \
--use_amp \
--use_module_replace \
--use_local_sgd \
--local_sgd_sync_interval 5 \
--local_sgd_warmup_steps 10 \
--clip_pseudo_grad 10 \
--gradnorm_weighted \
--skip_anomaly \
--skip_anomaly_warmup_steps 10 \
--outer_optim_class sgd
```

## 3D (Megatron-LM) Integration
EDiT can also be integrated with 3D training through megatron. In atorch, we provide such functionality through patches and ATorchTrainerV2.

ATorchTrainerV2 handles model construction and argument handling. To enable EDiT, we need to inject several special arguments to trainer's input arguments, and before everything, patch the megatron.

For the moment, local sgd is only tested for megatron 0.9.0

### Training

```python
from atorch.local_sgd.megatron import patch_megatron_for_local_sgd, local_sgd_args_provider

training_args.extra_configs["extra_args_provider"] = local_sgd_args_provider
training_args.extra_configs["use_local_sgd"] = True
training_args.extra_configs["local_sgd_sync_interval"] = 32
training_args.extra_configs["local_sgd_warmup_steps"] = 256

# then we have to patch megatron
patch_megatron_for_local_sgd()
```

### Example
```shell
cd examples/local_sgd/atorch_trainer_megatron

bash train.sh
```
Alternatively, you may reference to atorch/local_sgd/megatron, and migrate the patches directly into your Megatron-LM repo.
