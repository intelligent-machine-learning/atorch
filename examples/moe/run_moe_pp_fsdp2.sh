# 8 gpus at least
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)

STRATEGY_LIST="--use_fsdp2 --use_module_replace --use_checkpointing --moe_implementation_type MegaBlocks --moe_token_dispatcher_type AllToAll"

HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=2816
HEAD_NUM=16
KEY_VALUE_HEAD_NUM=4
LAYER_NUM=16
SEQ_LENGTH=4096
NUM_EXPERTS=64
NUM_SHARED_EXPERT=2
TOP_K=6

BATCHSIZE_PER_GPU=8
TRAIN_STEP=30
EP_SIZE=1

PP_STAGE_NUM=4
PP_NUM=4

EXTRA_PARAM=

# Loop through all the arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batchsize_per_gpu)
      BATCHSIZE_PER_GPU="$2"
      shift # Removes the current value of $1, making $2 become $1, $3 become $2, and so forth
      ;;
    --train_step)
      TRAIN_STEP="$2"
      shift # Removes the current value of $1, making $2 become $1, $3 become $2, and so forth
      ;;
    --max_checkpoint_module_num)
      EXTRA_PARAM="$EXTRA_PARAM --max_checkpoint_module_num $2"
      shift # Removes the current value of $1, making $2 become $1, $3 become $2, and so forth
      ;;
    --use_fp8)
	  EXTRA_PARAM="$EXTRA_PARAM --use_fp8"
      ;;
    --use_meta_init)
	  EXTRA_PARAM="$EXTRA_PARAM --use_meta_init"
      ;;
  esac
  shift # Move to the next argument in the list
done


echo HIDDEN_SIZE=$HIDDEN_SIZE
echo INTERMEDIATE_SIZE=$INTERMEDIATE_SIZE
echo HEAD_NUM=$HEAD_NUM
echo KEY_VALUE_HEAD_NUM=$KEY_VALUE_HEAD_NUM
echo LAYER_NUM=$LAYER_NUM
echo SEQ_LENGTH=$SEQ_LENGTH
echo BATCHSIZE_PER_GPU =  $BATCHSIZE_PER_GPU
echo TRAIN_STEP =  $TRAIN_STEP
echo EXTRA_PARAM is $EXTRA_PARAM
echo STRATEGY_LIST is $STRATEGY_LIST
echo EP_SIZE=$EP_SIZE
echo NUM_EXPERT=$NUM_EXPERTS
echo NUM_SHARED_EXPERT=$NUM_SHARED_EXPERT
echo TOP_K=$TOP_K


python -m atorch.distributed.run \
	--nproc_per_node="$NUM_GPUS_PER_NODE" \
	train_moe_with_pp.py $STRATEGY_LIST \
    --ep_size $EP_SIZE \
    --num_experts $NUM_EXPERTS \
    --num_shared_expert $NUM_SHARED_EXPERT \
    --top_k $TOP_K \
    --hidden_size $HIDDEN_SIZE \
    --intermediate_size $INTERMEDIATE_SIZE \
    --head_num $HEAD_NUM \
    --layer_num $LAYER_NUM \
    --seq_length $SEQ_LENGTH \
    --key_value_head_num $KEY_VALUE_HEAD_NUM \
    --max_train_step $TRAIN_STEP \
    --batchsize_per_gpu $BATCHSIZE_PER_GPU  $EXTRA_PARAM \
    --pp_stage_num $PP_STAGE_NUM --pp_num $PP_NUM \
    --pp_schedule 1F1B \
    --use_atorch_pp
