#!/bin/bash

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