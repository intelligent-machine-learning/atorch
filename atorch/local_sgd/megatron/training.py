# These are utility functions in Megatron, which take the cores and build the actual training system
# we patch all those necessary from args to model/optimizer to trainers
# ATorch megatron wrapper does not set up optimizer with setup_model_and_optimizer method, so no hack for it
# but NOTE it's necessary for native megatron training
# Since megatron wrapper has its customized training_log, we need to hack it there
import dataclasses

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config, is_float8tensor
from megatron.legacy.model import Float16Module
from megatron.training.global_vars import get_args

from .distributed_data_parallel import DistributedDataParallel as DDP


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and args.virtual_pipeline_model_parallel_size is not None:
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process)
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        if model_type == ModelType.encoder_and_decoder:
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                rank = mpu.get_pipeline_model_parallel_rank()
                first_decoder_rank = args.encoder_pipeline_model_parallel_size
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_inside_encoder(rank)
                add_decoder = mpu.is_inside_decoder(rank)
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    # The model_module.bfloat16()/model_module.half() above will call the inplace copy of TE's
    # Float8Tensor, which will write an unwanted value (amax calculated from the current fp8
    # param) to its amax_history. The following logic will correct the amax_history back.
    for model_module in model:
        for param in model_module.parameters():
            if is_float8tensor(param) and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta["scaling_fwd"]
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, "get_high_precision_init_val"):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(param.get_high_precision_init_val().abs().max())
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    if wrap_with_ddp:
        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs["grad_reduce_in_fp32"] = args.accumulate_allreduce_grads_in_fp32
        kwargs["check_for_nan_in_grad"] = args.check_for_nan_in_loss_and_grad
        kwargs["bucket_size"] = args.ddp_bucket_size
        kwargs["average_in_collective"] = args.ddp_average_in_collective
        ddp_config = DistributedDataParallelConfig(**kwargs)

        overlap_param_gather_with_optimizer_step = getattr(args, "overlap_param_gather_with_optimizer_step", False)
        extra_args = {}
        if args.use_local_sgd:
            from atorch.local_sgd import LocalSGDConfig

            local_sgd_config = LocalSGDConfig(
                local_sgd_sync_interval=args.local_sgd_sync_interval,
                local_sgd_warmup_steps=args.local_sgd_warmup_steps,
                clip_pseudo_grad=args.local_sgd_clip_pseudo_grad,
                skip_anomaly=args.local_sgd_skip_anomaly,
                ewma_alpha=args.local_sgd_ewma_alpha,
                ewma_warmup_steps=args.local_sgd_ewma_warmup_steps,
                ewma_threshold=args.local_sgd_ewma_threshold,
                pseudo_gradnorm_reduce=args.local_sgd_pseudo_gradnorm_reduce,
                weight_softmax_temperature=args.local_sgd_weight_softmax_temperature,
                cpu_offload=not args.on_device_local_sgd,
            )
            extra_args["local_sgd_config"] = local_sgd_config
        model = [
            DDP(
                config,
                ddp_config,
                model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step,
                **extra_args,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model
