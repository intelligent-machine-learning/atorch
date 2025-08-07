#!/bin/bash
set -exo pipefail

MEGATRON_PATH=/tmp/Megatron-LM
if [ ! -d ${MEGATRON_PATH} ]; then
    cd $(dirname ${MEGATRON_PATH})
    git clone -b core_r0.6.0 git@code.alipay.com:Arc/Megatron-LM.git
fi

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TIME_STAMP=$(date '+%Y%m%d-%H%M%S')
export OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/llama2_megatron/"}
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

NODE_NAME=${POD_NAME:-"master-0"}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [[ $POD_NAME =~ "edljob" ]]; then
    WORLD_SIZE=${WORKER_NUM:-1}
    NODE_RANK=${RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RANDOM_PORT=$[$RANDOM + 20000]
    MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
    GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
    echo "---> from edl runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}"
    LAUNCHER=" \
        python -m atorch.distributed.run --fault_tolerant --network-check \
        --max_restarts=1 \
        --nnode=$WORLD_SIZE \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_conf join_timeout=300 \
        "
else
    WORLD_SIZE=${WORLD_SIZE:-1}
    NODE_RANK=${RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RANDOM_PORT=$[$RANDOM + 20000]
    MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
    GPU_NUM=$((${GPUS_PER_NODE}*${WORLD_SIZE}))
    echo "---> from pytorch runtime, WORLD_SIZE: ${WORLD_SIZE}, NODE_RANK: ${NODE_RANK}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"
    LAUNCHER=" \
    torchrun \
    --nproc_per_node ${GPUS_PER_NODE} \
    --nnodes ${WORLD_SIZE} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    "
fi

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --distributed-backend nccl
    --use-distributed-optimizer
    --sequence-parallel
)

NETWORK_SIZE_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 32
    --ffn-hidden-size 11008
    --position-embedding-type rope
    --max-position-embeddings 4096
    --make-vocab-size-divisible-by 1
    --norm-epsilon 1.0e-5
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --use-flash-attn
)

LOGGING_ARGS=(
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --log-throughput
    --log-params-norm
)

REGULATIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 1.0e-1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 16
    --train-iters 10000
    --log-interval 1
    --tensorboard-dir ${OUTPUT_DIR}/runs/${TIME_STAMP}
    --disable-bias-linear
    --no-bias-gelu-fusion
    --optimizer adam
    --recompute-activations
    --recompute-granularity selective
    --use-mcore-models
    --transformer-impl transformer_engine
)

INITIALIZATION_ARGS=(
    --seed 1403
    --init-method-std 0.02
)
    
LEARNING_RATE_ARGS=(
    --lr 3.0e-5
    --lr-decay-style cosine
    --lr-warmup-fraction 0.1
    --min-lr 3.0e-6
)
    
CHECKPOINTING_ARGS=(
    --save ${OUTPUT_DIR}
    --save-interval 1000
)
    
MIXED_PRECISION_ARGS=(
    --bf16
)
    
VALIDATION_ARGS=(
    --eval-interval 2000
    --eval-iters 50
)

DATA_ARGS=(
    --data-path /hetero_infer/jinshi.cl/code/Megatron-LM-core_r0.6.0/wikitext-2-raw-v1-llama2/llama2_tokenized_train_text_document
    --split 949,50,1
    --seq-length 4096
    --num-workers 0
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model /hetero_infer/jinshi.cl/code/tokenizers/llama_tokenizer/tokenizer.model
    --data-cache-path ${OUTPUT_DIR}/data_cache
)

cd ${MEGATRON_PATH}
CMD="${LAUNCHER} \
       pretrain_gpt.py \
       ${DISTRIBUTED_ARGS[@]} \
       ${NETWORK_SIZE_ARGS[@]} \
       ${LOGGING_ARGS[@]} \
       ${REGULATIZATION_ARGS[@]} \
       ${TRAINING_ARGS[@]} \
       ${INITIALIZATION_ARGS[@]} \
       ${LEARNING_RATE_ARGS[@]} \
       ${CHECKPOINTING_ARGS[@]} \
       ${MIXED_PRECISION_ARGS[@]} \
       ${VALIDATION_ARGS[@]} \
       ${DATA_ARGS[@]} \
       "

echo ${CMD}
${CMD} 2>&1 | tee ${OUTPUT_DIR}/node_${NODE_RANK}_${TIME_STAMP}.log
