#!/bin/bash
set -exo pipefail

# GPU
# pip install deprecated tensorboard h5py

# PPU
# pip install scipy tensorboardX mdatasets nltk h5py

CONFIG_PATH=${1:-"$(dirname $0)/llama2_7b_config.yaml"}

MEGATRON_TAG=${MEGATRON_TAG:-ant_release_0.11.0_v1.6.0}
MEGATRON_PATH=/tmp/Megatron-LM-${MEGATRON_TAG}
if [ ! -d ${MEGATRON_PATH} ]; then
    pushd $(dirname ${MEGATRON_PATH})
    git clone -b ${MEGATRON_TAG} https://code.alipay.com/Arc/Megatron-LM.git $(basename ${MEGATRON_PATH})
    popd
fi

export PYTHONPATH="$(dirname $0)/../../../":${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TIME_STAMP=$(date '+%Y%m%d-%H%M%S')
export OUTPUT_DIR=${OUTPUT_DIR:-"/tmp/llama2_atorch_trainer/"}
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

CMD="${LAUNCHER[@]} $(dirname $0)/pretrain_atorch_trainer_megatron.py ${CONFIG_PATH}"

echo ${CMD}
${CMD} 2>&1 | tee ${OUTPUT_DIR}/node_${NODE_RANK}_${TIME_STAMP}.log
