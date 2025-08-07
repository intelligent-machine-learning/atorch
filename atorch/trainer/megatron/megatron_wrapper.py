"""
Megatron wrapper.
"""
import dataclasses
import gc
import inspect
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from dependency_injector.wiring import Provide, inject
from packaging.version import Version
from torch.utils.data import DataLoader
from transformers.trainer import TRAINER_STATE_NAME

from atorch.common.log_utils import default_logger as logger
from atorch.common.log_utils import log_rank_0
from atorch.trainer.args import AtorchTrainingArgs, MegatronArgs
from atorch.trainer.base.atorch_container import AtorchTrainerContainer
from atorch.trainer.base.atorch_train_engine import AtorchTrainEngine
from atorch.trainer.megatron.megatron_ckpt_loader import MegatronCkptLoader
from atorch.trainer.megatron.megatron_ckpt_saver import MegatronCkptSaver
from atorch.trainer.megatron.megatron_dataloader import (
    AtorchMegatronDataloader,
    _prepare_megaton_dataloader,
    wrap_megatron_dataloader,
)
from atorch.trainer.megatron.megatron_train_step import BertTrainStep, GPTTrainStep, MegatronTrainStep, T5TrainStep
from atorch.trainer.trainer_callback import AtorchTrainerCallback, AtorchTrainerControl, AtorchTrainerState
from atorch.trainer.utils import (
    broadcast_spike_loss_ratio_in_pp_group,
    calc_params_std,
    count_zeros_in_grad,
    get_grad_norm_in_optimizer,
    scale_main_grad_for_spike_loss,
    training_log,
)
from atorch.utils.import_util import is_megatron_lm_available, is_torch_npu_available
from atorch.utils.version import get_megatron_version, is_megatron_version_bigger_than
from atorch.utils.virtual_optimizer.megatron_virtual_optimizer import get_megatron_virtual_optimizer

if is_megatron_lm_available():
    from megatron.core import mpu, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig, get_megatron_optimizer
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.utils import get_model_config
    from megatron.legacy.model import BertModel, GPTModel, T5Model
    from megatron.legacy.model.classification import Classification
    from megatron.legacy.model.module import MegatronModule
    from megatron.training import get_args, get_tensorboard_writer, initialize_megatron, print_rank_last

    try:
        from megatron.training import get_num_microbatches, update_num_microbatches
    except ImportError:
        from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
    from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
    from megatron.training.checkpointing import (
        checkpoint_exists,
        load_args_from_checkpoint,
        load_checkpoint,
        save_checkpoint,
    )
    from megatron.training.global_vars import get_timers, set_global_variables
    from megatron.training.initialize import (
        _compile_dependencies,
        _init_autoresume,
        _initialize_distributed,
        _initialize_tp_communicators,
        _set_random_seed,
        set_jit_fusion_options,
        write_args_to_tensorboard,
    )

    try:
        from megatron.training.optimizer_param_scheduler import OptimizerParamScheduler
    except ImportError:
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.training.training import (
        build_train_valid_test_data_iterators,
        get_optimizer_param_scheduler,
        num_floating_point_operations,
    )
    from megatron.training.utils import calc_params_l2_norm, unwrap_model
    from megatron.training.yaml_arguments import validate_yaml


DATALOADER_INDEX_MAPPER = dict(train=0, eval=1, test=2)


def setup_args(extra_args_provider=None, args_defaults={}, ignore_unknown_args=False):
    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)

    # Set defaults
    for key, value in args_defaults.items():
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print(
                    f"WARNING: overriding default arguments for " f"{key}:{getattr(args, key)} with {key}:{value}",
                    flush=True,
                )
        # set extra_configs.
        setattr(args, key, value)

    return args


def set_deterministic_algorithms(args):
    """
    args: Megatron args, acquired by get_args()
    """
    # Megatron's `deterministic_mode` arg is introduced after core_0.8.0 version.
    if is_megatron_version_bigger_than("0.8.0"):
        args.deterministic_mode = True
        args.use_flash_attn = False
        args.cross_entropy_loss_fusion = False

    if not is_torch_npu_available():
        # On GPU env, bias_dropout_fusion will effect the accuracy of loss and grad when resuming
        # training from a checkpoint. So if you want to use deterministic algorithms, set it to False.
        args.bias_dropout_fusion = False

    # Set env variables about deterministic mode
    if os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") != "0":
        logger.info("For deterministic algo, env [NVTE_ALLOW_NONDETERMINISTIC_ALGO] will be set to '0'.")
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
    if os.getenv("NCCL_ALGO") not in all_reduce_choices:
        logger.info("For deterministic algo, env [NCCL_ALGO] will be set to 'Ring'.")
        os.environ["NCCL_ALGO"] = "Ring"

    cublas_workspace_config_choices = [":4096:8", ":16:8"]
    if os.getenv("CUBLAS_WORKSPACE_CONFIG") not in cublas_workspace_config_choices:
        logger.info("For deterministic algo, env [CUBLAS_WORKSPACE_CONFIG] will be set to ':4096:8'.")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.use_deterministic_algorithms(True, warn_only=True)


# Deprecated! initialize_megatron_legacy() is deprecated when megatron >= 0.12
def initialize_megatron_legacy(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    get_output_layer_ranks=None,
    use_deterministic_algorithms=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """

    assert not is_megatron_version_bigger_than(
        "0.12.0"
    ), "Please call original initialize_megatron() under Megatron 0.12!"

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    args = setup_args(extra_args_provider, args_defaults, ignore_unknown_args)

    if use_deterministic_algorithms:
        set_deterministic_algorithms(args)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoint-args requires --load argument"
        assert getattr(args, "non_persistent_ckpt_type", None) != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # set logging level
    if is_megatron_version_bigger_than("0.8.0"):
        from megatron.training.initialize import setup_logging

        setup_logging()

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.

        if not is_megatron_version_bigger_than("0.9.0"):
            _initialize_distributed()
        elif "get_output_layer_ranks" in inspect.signature(_initialize_distributed).parameters:
            _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks, get_output_layer_ranks)
        else:
            _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))

        if is_megatron_version_bigger_than("0.11.0"):
            _set_random_seed(args.seed, args.data_parallel_random_init, args.te_rng_tracker, args.inference_rng_tracker)
        else:
            _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def prepare_optimizer(train_args: AtorchTrainingArgs, megatron_args: MegatronArgs, model):
    logger.info("Preparing optimizer")
    args = get_args()
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = get_timers()
    # NOTE use_local_sgd should be injected with local_sgd_arg_provider
    if hasattr(args, "use_local_sgd") and args.use_local_sgd:
        if get_megatron_version() != Version("0.9.0"):
            logger.warning("[WARNING!!!!!!] local sgd is only tested under Megatron 0.9.0 !!!")

        from atorch.local_sgd.configs import GTAConfig, LocalSGDConfig, OuterOptimizerConfig
        from atorch.local_sgd.megatron import get_megatron_optimizer as get_megatron_optimizer_local_sgd

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
        outer_optim_config = OuterOptimizerConfig(
            outer_optim_class=torch.optim.SGD if args.outer_optimizer == "sgd" else None,
            outer_optim_kwargs={
                "lr": args.outer_optimizer_lr,
                "momentum": args.outer_optimizer_momentum,
                "nesterov": args.outer_optimizer_nesterov if args.outer_optimizer_momentum != 0.0 else False,
            },
        )
        gta_config = GTAConfig(
            reducer=args.local_sgd_pseudo_grad_reducer,
            consensus_method=None if args.local_sgd_pseudo_grad_reducer != "gta" else "count",
            sparsification_method=None,
            normalize=args.local_sgd_pseudo_grad_normalize,
            density=1.0,
            int8_mask=None,
        )
        return get_megatron_optimizer_local_sgd(
            config,
            model,
            megatron_args.no_wd_decay_cond,
            megatron_args.scale_lr_cond,
            megatron_args.lr_mult,
            local_sgd_config=local_sgd_config,
            outer_optim_config=outer_optim_config,
            gta_config=gta_config,
        )
    else:
        if train_args.use_virtual_optimizer:
            return get_megatron_virtual_optimizer(
                train_args,
                config,
                model,
                megatron_args.no_wd_decay_cond,
                megatron_args.scale_lr_cond,
                megatron_args.lr_mult,
            )
        else:
            return get_megatron_optimizer(
                config,
                model,
                megatron_args.no_wd_decay_cond,
                megatron_args.scale_lr_cond,
                megatron_args.lr_mult,
            )


def model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the model."""
    args = get_args()
    mode = "pre-training" if args.pretraining_flag else "fine-tuning"
    if args.rank == 0:
        print(f"Building {args.model_type_name} model in the {mode} mode.")
        print(
            "The Megatron LM model weights are initialized at random in `accelerator.prepare`. "
            "Please use `accelerator.load_checkpoint` to load a pre-trained checkpoint matching the distributed setup."
        )
    config = core_transformer_config_from_args(args)
    if args.model_type_name == "bert":
        if args.pretraining_flag:
            num_tokentypes = 2 if args.bert_binary_head else 0
            model = BertModel(
                config=config,
                num_tokentypes=num_tokentypes,
                add_binary_head=args.bert_binary_head,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
        else:
            model = Classification(
                config=config,
                num_classes=args.num_labels,
                num_tokentypes=2,
                pre_process=pre_process,
                post_process=post_process,
            )
    elif args.model_type_name == "gpt":
        model = GPTModel(
            config=config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    elif args.model_type_name == "t5":
        model = T5Model(
            config=config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type_name}")
    return model


# override megatron/training/training::should_disable_forward_pre_hook() in Megatron (>=0.12)
def should_disable_forward_pre_hook():
    """Block forward pre-hook for certain configurations."""
    args = get_args()
    return not getattr(args, "use_custom_fsdp", False) and args.use_distributed_optimizer and args.overlap_param_gather


class AtorchMegatronEngine(AtorchTrainEngine):
    @inject
    def __init__(
        self,
        train_args: AtorchTrainingArgs,
        dataloaders: Union[AtorchMegatronDataloader, tuple, None],
        train_state: AtorchTrainerState,
        ckpt_saver: MegatronCkptSaver = Provide[AtorchTrainerContainer.ckpt_saver],
        ckpt_loader: MegatronCkptLoader = Provide[AtorchTrainerContainer.ckpt_loader],
        resume_from_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(train_args, train_state)

        self.module: List[MegatronModule]
        self.optimizer: MegatronOptimizer
        self.scheduler: OptimizerParamScheduler
        self.dataloaders = dataloaders

        self.ckpt_saver: MegatronCkptSaver = ckpt_saver
        self.ckpt_loader: MegatronCkptLoader = ckpt_loader

        self._dataloaders: List[DataLoader] = []

        self.megatron_args = train_args.megatron_args()

        self.initialize()

        self.num_floating_point_operations_since_last_log_event = 0.0

        self._validate_train_args()

        # Set megatron args
        self.megatron_args.num_micro_batches = get_num_microbatches()

        args = get_args()

        # TODO: Add pretraining_flag arg to AtorchTrainingArgs to decide to launch pretrain or finetune.
        args.pretraining_flag = True

        # Just for resuming from checkpoint
        # Record the rng state loaded from checkpoint, which will be restored after building dataloader
        self.rng_state_from_ckpt = None

        # If in finetune, dataloader should be built before the scheduler is created.
        # `train_iters` is needed when creating scheduler. However, it is not set in finetune scene usually,
        # calculated when building dataloader instead.
        delay_building_dataloader = self.train_args.finetune_type is None

        # Building dataloader will not be executed when just converting checkpoint.
        if self.train_args.convert_checkpoint:
            delay_building_dataloader = True

            # `train_iters` or `train_samples` should be set before creating scheduler.
            if args.train_iters is None and args.train_samples is None:
                args.train_iters = 1

        if not delay_building_dataloader:
            self.build_dataloader()

        # This invoking must be executed when initializing megatron has been down.
        if getattr(args, "swap_attention", None):
            try:
                from mindspeed.core.memory.adaptive_recomputing.adaptive_recompute import (
                    setup_model_and_optimizer_wrapper,
                )

                setattr(
                    AtorchMegatronEngine,
                    "prepare_model_optimizer_scheduler",
                    setup_model_and_optimizer_wrapper(AtorchMegatronEngine.prepare_model_optimizer_scheduler),
                )
            except Exception:
                raise ValueError("not support swap_attention")

        timers = get_timers()

        # Model, optimizer, and learning rate.
        timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
        (
            self.module,
            self.optimizer,
            self.scheduler,
        ) = self.prepare_model_optimizer_scheduler(resume_from_checkpoint=resume_from_checkpoint)
        timers("model-and-optimizer-setup").stop()

        if self.train_args.convert_checkpoint:
            args.save = self.train_args.output_dir_for_converting
            args.async_save = False  # asynchronous save is not supported when converting checkpoint

            if is_megatron_version_bigger_than("0.9.0"):
                from megatron.training.utils import update_use_dist_ckpt

                update_use_dist_ckpt(args)

            if is_megatron_version_bigger_than("0.10.0"):
                from megatron.training.training import preprocess_common_state_dict

                save_checkpoint(
                    args.iteration,
                    self.module,
                    self.optimizer,
                    self.scheduler,
                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far,
                    preprocess_common_state_dict_fn=preprocess_common_state_dict,
                )
            else:
                save_checkpoint(
                    args.iteration,
                    self.module,
                    self.optimizer,
                    self.scheduler,
                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far,
                )

            torch.distributed.barrier()

            sys.exit()

        self.monitor = None
        if self.train_args.msprob_monitor_config_file is not None:
            from msprobe.pytorch import TrainerMon

            self.monitor = TrainerMon(
                config_file_path=self.train_args.msprob_monitor_config_file,
                process_group=None,
                params_have_main_grad=True,
                opt_ty=None,
            )

            self.monitor.set_wrapped_optimizer(self.optimizer)

            self.monitor.monitor_gnorm_with_ad(
                self.module,
                grad_acc_steps=get_num_microbatches(),
                optimizer=None,
                dp_group=None,
                tp_group=None,
            )

        # Moe_routing_map save and load
        self.MoELayerDict: Dict[str, MoELayer] = {}
        self.saveIter: List[int] = []

        # TODO: define a function to unify barrier operator.
        torch.distributed.barrier()

        args.model_return_dict = None
        if self.megatron_args.custom_train_step_class is not None:
            if self.megatron_args.custom_train_step_kwargs is None:
                self.megatron_args.custom_train_step_kwargs = {}
            self.train_step_handler = self.megatron_args.custom_train_step_class(
                args, **self.megatron_args.custom_train_step_kwargs
            )
            assert isinstance(self.train_step_handler, MegatronTrainStep)
        elif args.model_type_name == "bert":
            self.train_step_handler = BertTrainStep(args)
        elif args.model_type_name == "gpt":
            self.train_step_handler = GPTTrainStep(args)
        elif args.model_type_name == "t5":
            self.train_step_handler = T5TrainStep(args)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type_name}")
        self.optimizer.skipped_iter = False

        # Tracking loss.
        self.total_loss_dict = {}  # type: ignore[var-annotated]
        self.eval_total_loss_dict = {}  # type: ignore[var-annotated]

        self.report_memory_flag = True
        self.pre_hook_enabled = False
        self.num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
        self.module_config = None
        self.training_log_args = None
        self.custom_training_log_dict = None
        self.num_microbatches = get_num_microbatches()
        self.skipped_iter = 0

        if delay_building_dataloader:
            self.build_dataloader()

        # Restore rng state when resuming checkpoint
        if self.rng_state_from_ckpt is not None:
            logger.info(
                f"[Rank {torch.distributed.get_rank()}] restore random number generator state after "
                "building dataloader."
            )
            random.setstate(self.rng_state_from_ckpt["random_rng_state"])
            np.random.set_state(self.rng_state_from_ckpt["np_rng_state"])
            torch.set_rng_state(self.rng_state_from_ckpt["torch_rng_state"])
            torch.cuda.set_rng_state(self.rng_state_from_ckpt["cuda_rng_state"])
            tensor_parallel.get_cuda_rng_tracker().set_states(self.rng_state_from_ckpt["rng_tracker_states"])

        # set train_args
        self.train_args.per_device_train_batch_size = args.micro_batch_size * get_num_microbatches()
        self.train_args.per_device_eval_batch_size = (
            args.global_batch_size // args.data_parallel_size
        )  # Don't consider batch size warmup

        write_args_to_tensorboard()

        # Print setup timing.
        log_rank_0("done with setup ...")
        timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)

    def _validate_train_args(
        self,
    ):
        args = get_args()

        if args.profile:
            raise ValueError(
                "Please set atorch trainer's 'profiler_type' arg, megatron profiler is disabled by" "atorch trainer."
            )

        def _assert(condition, feature, recommend_key_value: dict = None):
            err_info = f"{feature} is not supported in AtorchTrainerV2."
            if recommend_key_value is not None:
                arg_name = list(recommend_key_value.keys())[0]
                err_info += f" Please set {arg_name} to {recommend_key_value[arg_name]} ."
            assert condition, err_info

        _assert(not args.enable_one_logger, "one_logger", {"enable_one_logger": False})
        _assert(not args.adlr_autoresume, "autoresume on adlr cluster", {"adlr_autoresume": False})
        _assert(not args.exit_signal_handler, "exit_signal_handler", {"exit_signal_handler": False})
        _assert(args.exit_duration_in_mins is None, "exit_duration_in_mins", {"exit_duration_in_mins": None})
        _assert(args.exit_interval is None, "exit_interval", {"exit_interval": None})
        _assert(not args.vision_pretraining, "vision_pretraining", {"vision_pretraining": False})
        _assert(
            not getattr(args, "enable_ft_package", False),
            "NVIDIA Fault Tolerance",
            {"enable_ft_package": False},
        )
        _assert(
            getattr(args, "non_persistent_ckpt_type", None) is None,
            "non-persistent model checkpoints",
            {"non_persistent_ckpt_type": None},
        )
        _assert(not args.log_progress, "log_progress", {"log_progress": False})
        _assert(
            not getattr(args, "run_workload_inspector_server", False),
            "workload inspector",
            {"run_workload_inspector_server": False},
        )
        _assert(not getattr(args, "log_straggler", False), "StragglerDetector", {"log_straggler": False})
        _assert(len(getattr(args, "iterations_to_skip", [])) == 0, "iterations_to_skip", {"iterations_to_skip": []})
        _assert(
            not getattr(args, "decrease_batch_size_if_needed", False),
            "decrease batch size",
            {"decrease_batch_size_if_needed": False},
        )
        _assert(
            getattr(args, "train_sync_interval", None) is None,
            "Training CPU-GPU synchronization interval",
            {"train_sync_interval": None},
        )
        # TODO: to support check_weight_hash_across_dp_replicas_interval
        _assert(
            getattr(args, "check_weight_hash_across_dp_replicas_interval", None) is None,
            "check_weight_hash_across_dp_replicas_interval",
            {"check_weight_hash_across_dp_replicas_interval": None},
        )

        assert (
            args.ckpt_convert_format is None
        ), "'ckpt_convert_format' is not supported in AtorchTrainer, please use AtorchTrainer's 'convert_checkpoint' instead."  # noqa E501

    def initialize(self):
        # If megatron >= 0.12.0, call original initialize_megatron() function of Megatron
        if is_megatron_version_bigger_than("0.12.0"):
            # set up args
            parsed_args = setup_args(
                extra_args_provider=self.megatron_args.extra_args_provider,
                args_defaults=self.megatron_args.to_dict(),
                ignore_unknown_args=True,
            )

            if self.train_args.use_deterministic_algorithms:
                set_deterministic_algorithms(parsed_args)

            if "get_output_layer_ranks" in inspect.signature(_initialize_distributed).parameters:
                # Megatron EA version
                initialize_megatron(
                    extra_args_provider=self.megatron_args.extra_args_provider,
                    args_defaults=self.megatron_args.to_dict(),
                    ignore_unknown_args=True,
                    get_embedding_ranks=self.megatron_args.get_embedding_ranks,
                    get_position_embedding_ranks=self.megatron_args.get_position_embedding_ranks,
                    get_output_layer_ranks=self.megatron_args.get_output_layer_ranks,
                    parsed_args=parsed_args,
                )
            else:
                initialize_megatron(
                    extra_args_provider=self.megatron_args.extra_args_provider,
                    args_defaults=self.megatron_args.to_dict(),
                    ignore_unknown_args=True,
                    get_embedding_ranks=self.megatron_args.get_embedding_ranks,
                    get_position_embedding_ranks=self.megatron_args.get_position_embedding_ranks,
                    parsed_args=parsed_args,
                )
        else:
            initialize_megatron_legacy(
                extra_args_provider=self.megatron_args.extra_args_provider,
                args_defaults=self.megatron_args.to_dict(),
                ignore_unknown_args=True,
                get_embedding_ranks=self.megatron_args.get_embedding_ranks,
                get_position_embedding_ranks=self.megatron_args.get_position_embedding_ranks,
                get_output_layer_ranks=self.megatron_args.get_output_layer_ranks,
                use_deterministic_algorithms=self.train_args.use_deterministic_algorithms,
            )

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        if self.train_args.use_deterministic_algorithms:
            TORCH_MAJOR = int(torch.__version__.split(".")[0])
            TORCH_MINOR = int(torch.__version__.split(".")[1])
            if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
                torch._C._jit_set_nvfuser_enabled(False)

    def prepare_model_optimizer_scheduler(self, resume_from_checkpoint=None):
        logger.info("Preparing model optimizer scheduler")
        args = get_args()
        timers = get_timers()
        # TODO: remove some redundant code.
        if self.megatron_args.custom_prepare_model_function is not None:
            if self.megatron_args.custom_model_provider_function is None:
                raise ValueError(
                    "You must provide a `custom_model_provider_function` when using a `custom_prepare_model_function`."
                )
            model_provider_func_ = self.megatron_args.custom_model_provider_function
            model = self.megatron_args.custom_prepare_model_function(model_provider_func_)
        else:
            model_type = ModelType.encoder_or_decoder
            if args.model_type_name == "t5":
                model_type = ModelType.encoder_and_decoder
            if self.megatron_args.custom_model_provider_function is not None:
                model_provider_func_ = self.megatron_args.custom_model_provider_function
            else:
                model_provider_func_ = model_provider_func
            # NOTE similar to get optimizer, hack here, and need to verify the existance of use_local_sgd
            if hasattr(args, "use_local_sgd") and args.use_local_sgd:
                from atorch.local_sgd.megatron import get_model

                model = get_model(model_provider_func_, model_type)
            else:
                from megatron.training.training import get_model

                model = get_model(model_provider_func_, model_type)

        unwrapped_model = unwrap_model(model)

        optimizer = prepare_optimizer(self.train_args, self.megatron_args, model)
        scheduler = get_optimizer_param_scheduler(optimizer)

        try:
            from ant_utils.dcp_utils.dcp_utils import patch_torch

            patch_torch(patch_find_nd_overlapping_shards=self.train_args.patch_find_nd_overlapping_shards)
        except ImportError:
            logger.warning(
                "Unable to import patch_torch from ant_utils.dcp_utils.dcp_utils, you might not using "
                "available megatron version. If you want to use a speedup version of megatron,"
                "please read atorch doc to use a proper megatron version."
            )

        if is_megatron_version_bigger_than("0.9.0") and args.moe_use_upcycling:
            from megatron.core.transformer.moe import upcycling_utils

            torch.distributed.barrier()
            assert not checkpoint_exists(args.save), (
                "The upcycling destination directory already exists. "
                "Please check if --moe-use-upcycling is mistakenly enabled. "
                "Upcycling should only be set for the first run when converting the dense model. "
                "All subsequent runs should remove this flag. "
            )
            num_experts = args.num_experts
            args.num_experts = None
            expert_model_parallel_size = args.expert_model_parallel_size
            args.expert_model_parallel_size = 1
            dense_model_for_upcycling = get_model(model_provider_func_, model_type)
            args.num_experts = num_experts
            args.expert_model_parallel_size = expert_model_parallel_size
            _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
                load_checkpoint,
                unwrapped_model,
                dense_model_for_upcycling,
                load_kwargs={"model": dense_model_for_upcycling, "optimizer": None, "opt_param_scheduler": None},
            )
            args.iteration = 1
            save_checkpoint(args.iteration, model, None, None, args.num_floating_point_operations_so_far)
            torch.distributed.barrier()
            del dense_model_for_upcycling
            if (args.fp16 or args.bf16) and optimizer is not None:
                optimizer.reload_model_params()
            log_rank_0(f"Upcycled checkpoint saved to {args.save}")

        if resume_from_checkpoint is not None and not (
            is_megatron_version_bigger_than("0.9.0") and args.moe_use_upcycling
        ):
            timers("load-checkpoint", log_level=0).start(barrier=True)

            if isinstance(resume_from_checkpoint, str):
                resume_from_checkpoint = Path(resume_from_checkpoint)
            (args.iteration, args.num_floating_point_operations_so_far,) = self.load_checkpoint(
                resume_from_ckpt=resume_from_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            timers("load-checkpoint").stop(barrier=True)
            timers.log(["load-checkpoint"])

            if args.iteration > 0 and not args.finetune and not args.no_load_rng:
                # Record rng state after loading checkpoint
                self.rng_state_from_ckpt = {
                    "random_rng_state": random.getstate(),
                    "np_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                    "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
                }
        else:
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0

        # get model without FP16 and/or DDP wrappers
        if (
            args.iteration == 0
            and len(unwrapped_model) == 1
            and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
        ):
            log_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        self.iteration = args.iteration
        self.start_iteration = args.iteration
        self.num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
        args.global_step = args.iteration

        args.model_len = len(model)
        return model, optimizer, scheduler

    def get_dataloader(self, name=None):
        idx = DATALOADER_INDEX_MAPPER.get(name, None)
        assert idx is not None, f"dataloader name {name} is unknown, please use 'train','eval' or 'test'"

        try:
            return self._dataloaders[idx]
        except IndexError:
            raise ValueError(
                f"dataloader {name} is not found in Megatron engine, please make sure you have configured that"
            )

    def build_dataloader(self):
        def _build_dataloader(vp_stage=None):
            if self.megatron_args.custom_megatron_dataloaders_provider_function is not None:
                if (
                    "vp_stage"
                    in inspect.signature(self.megatron_args.custom_megatron_dataloaders_provider_function).parameters
                ):
                    (
                        train_data_iterator,
                        valid_data_iterator,
                        test_data_iterator,
                    ) = self.megatron_args.custom_megatron_dataloaders_provider_function(vp_stage=vp_stage)
                else:
                    (
                        train_data_iterator,
                        valid_data_iterator,
                        test_data_iterator,
                    ) = self.megatron_args.custom_megatron_dataloaders_provider_function()
                return train_data_iterator, valid_data_iterator, test_data_iterator
            elif self.megatron_args.custom_megatron_datasets_provider_function is not None:
                (
                    train_data_iterator,
                    valid_data_iterator,
                    test_data_iterator,
                ) = build_train_valid_test_data_iterators(self.megatron_args.custom_megatron_datasets_provider_function)
                return train_data_iterator, valid_data_iterator, test_data_iterator
            elif self.dataloaders is not None:
                logger.warning("It has not been tested enough!")
                (
                    train_data_iterator,
                    valid_data_iterator,
                    test_data_iterator,
                ) = _prepare_megaton_dataloader(self.train_args, self.dataloaders)
                # self._dataloaders.extend(_prepare_megaton_dataloader(self.train_args, self.dataloaders))

        is_post_training = self.train_args.finetune_type is not None

        args = get_args()
        timers = get_timers()

        timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)

                if is_post_training:
                    vp_stage = None
                else:
                    unwrapped_model = unwrap_model(self.module[i])
                    vp_stage = unwrapped_model.vp_stage if hasattr(unwrapped_model, "vp_stage") else None

                iterators = _build_dataloader(vp_stage=vp_stage)
                train_data_iterator.append(iterators[0])
                valid_data_iterator.append(iterators[1])
                test_data_iterator.append(iterators[2])
        else:
            (
                train_data_iterator,
                valid_data_iterator,
                test_data_iterator,
            ) = _build_dataloader()
        timers("train/valid/test-data-iterators-setup").stop()

        if is_post_training:
            # In post-training scene, we need to calculate how many global steps are required to iterate through the
            # dataloader. Batch size warmup will increase the complexity of calculating the steps, so we don't consider
            # batch size warmup in post-training scene.
            assert args.rampup_batch_size is None, "rampup_batch_size is not supported in post-training."

        self._dataloaders.append(
            wrap_megatron_dataloader(train_data_iterator, dataset_type="train", is_post_training=is_post_training)
        )
        self._dataloaders.append(
            wrap_megatron_dataloader(valid_data_iterator, dataset_type="eval", is_post_training=is_post_training)
        )
        self._dataloaders.append(
            wrap_megatron_dataloader(test_data_iterator, dataset_type="test", is_post_training=is_post_training)
        )

        # Calculate train_iters
        if is_post_training:
            train_dataloader = self._dataloaders[0]
            num_update_steps_per_epoch = len(train_dataloader)

            if (
                self.train_args.max_steps > 0
                and args.train_iters is not None
                and self.train_args.max_steps != args.train_iters
            ):
                logger.warning("trainer args.max_steps will be overwritten by megatron_args.train_iters.")
                self.train_args.max_steps = args.train_iters

            if self.train_args.max_steps < 0:
                self.train_args.max_steps = math.ceil(self.train_args.num_train_epochs * num_update_steps_per_epoch)

            args.train_iters = self.train_args.max_steps

        torch.distributed.barrier()

    def train(self):
        for model_module in self.module:
            model_module.train()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

    def forward(self, data_iterator):
        # During training, we use train_step()
        # model(**batch_data) performs following operations by delegating it to `self.train_step`:
        # 1. Prepare **batch_data for Tendor, Pipeline and Model Parallelism
        # 2. Set grad to zero.
        # 3. forward pass and backward pass using Pipeline Parallelism
        # 4. Empty unused memory.
        # 5. Reduce gradients.
        # 6. Update parameters.
        # 7. Gather params when using Distributed Optimizer (Data Parallelism).
        # 8. Update learning rate if scheduler is specified.
        # 9. Empty unused memory.
        # 10. Average loss across microbatches and across DP ranks.
        #
        # During evaluation, we use eval_step()
        args = get_args()

        if self.module[0].training:
            args.forward_mode = "train"
            # Update number of microbatches first without consistency. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False, verbose=True)
            if get_num_microbatches() != self.num_microbatches and self.iteration != 0:
                assert (
                    get_num_microbatches() > self.num_microbatches
                ), "number of microbatches should be increasing due to batch size rampup"
            self.num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True, verbose=True)
            args.curr_iteration = self.iteration
            loss_dict, self.skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(data_iterator)

            self.iteration += 1

            # save routing map for moe-layer
            if getattr(args, "moe_router_save", False):
                for router_save_iter in self.saveIter:
                    if self.iteration == router_save_iter:
                        for path, layer in self.MoELayerDict.items():
                            layer.save_routing_map(
                                router_save_iter, path, args.moe_router_save_dir, args.moe_splits_save_dir
                            )

            args.global_step = self.iteration
            batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            args.consumed_train_samples += batch_size
            num_floating_point_operations_in_batch = num_floating_point_operations(args, batch_size)
            self.num_floating_point_operations_so_far += num_floating_point_operations_in_batch
            self.num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

            if getattr(self.optimizer, "is_stub_optimizer", False):
                loss_scale = 1.0
            else:
                loss_scale = self.optimizer.get_loss_scale().item()

            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(self.module)
            params_std = None
            if self.train_args.log_params_std:
                params_std = calc_params_std(self.module, gather_to_last_rank=True)
            custom_training_log_dict = None
            if self.megatron_args.custom_tensorboard_record_calculate_fn is not None:
                # Just for custom metrics.
                args.grad_norm = grad_norm
                custom_tensorboard_record_calculate_fn = self.megatron_args.custom_tensorboard_record_calculate_fn
                custom_training_log_dict = custom_tensorboard_record_calculate_fn(self.module)

            learning_rate = None
            decoupled_learning_rate = None
            for param_group in self.optimizer.param_groups:
                if param_group["is_decoupled_lr"]:
                    decoupled_learning_rate = param_group["lr"]
                else:
                    learning_rate = param_group["lr"]
            self.training_log_args = [
                loss_dict,
                self.total_loss_dict,
                learning_rate,
                decoupled_learning_rate,
                self.iteration,
                loss_scale,
                self.report_memory_flag,
                self.skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
                params_std,
                custom_training_log_dict,
            ]
        else:
            # Set evaluation type, which belongs to ["eval", "test"]
            args.forward_mode = self.train_args.eval_type
            loss_dict = self.eval_step(data_iterator)
            for key, value in loss_dict.items():
                if self.train_args.eval_type == "eval":
                    key = f"{key} validation"
                elif self.train_args.eval_type == "test":
                    key = f"{key} test"
                else:
                    logger.warning(f"Unexpected evaluation type {self.train_args.eval_type}")
                self.eval_total_loss_dict[key] = (
                    self.eval_total_loss_dict.get(key, torch.FloatTensor([0.0]).cuda()) + value
                )
                self.eval_total_loss_dict[key + "_num_iters"] = (
                    self.eval_total_loss_dict.get(key + "_num_iters", torch.FloatTensor([0.0]).cuda())
                    + torch.FloatTensor([1.0]).cuda()
                )

        loss = torch.tensor(0.0, device=torch.cuda.current_device())
        for key in loss_dict:
            if len(loss_dict[key].shape) == 0:
                loss += loss_dict[key]

        logits = None
        if "logits" in loss_dict:
            logits = loss_dict["logits"]

        # model_output_class: Return a object like transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        if self.train_step_handler.model_output_class is not None:
            return self.train_step_handler.model_output_class(loss=loss, logits=logits)
        return loss

    def train_step(self, data_iterator):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """

        args = get_args()
        timers = get_timers()

        self.optimizer_zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        # losses_reduced: A list of loss whose length equals to the number of microbatches.
        losses_reduced = forward_backward_func(
            forward_step_func=self.train_step_handler.get_forward_step_func(),
            data_iterator=data_iterator,
            model=self.module,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )

        total_grad_norm = None

        if self.train_args.monitor_total_grad_norm:
            total_grad_norm = get_grad_norm_in_optimizer(self.optimizer)

        monitored_dict = {"losses_reduced": losses_reduced, "total_grad_norm": total_grad_norm}

        loss_postprocessed = self.train_step_handler.loss_postprocessing(monitored_dict)

        if isinstance(loss_postprocessed, dict):
            if "loss_to_log" in loss_postprocessed:
                assert (
                    "spike_loss_ratio" in loss_postprocessed
                ), f"'spike_loss_ratio' should be in return value of {type(self.train_step_handler)}::loss_postprocessing()."  # noqa E501
                loss_to_log = loss_postprocessed["loss_to_log"]
                spike_loss_ratio = loss_postprocessed["spike_loss_ratio"]
            else:  # compat atorch version before 1.6.1rc8
                loss_to_log = loss_postprocessed
                spike_loss_ratio = None
        elif isinstance(loss_postprocessed, tuple):  # compat atorch version before 1.6.1rc8
            assert len(loss_postprocessed) == 2
            loss_to_log, spike_loss_ratio = loss_postprocessed
        else:
            raise ValueError(
                f"Unexpected return value type of {type(self.train_step_handler)}::loss_postprocessing()."
                f" Please use dict to wrap return value of the loss_postprocessing method."
            )

        spike_loss_ratio = broadcast_spike_loss_ratio_in_pp_group(spike_loss_ratio)

        if spike_loss_ratio is not None:
            logger.info(f"[Rank {torch.distributed.get_rank()}] apply spike loss on grad with ratio {spike_loss_ratio}")
            scale_main_grad_for_spike_loss(
                self.module,
                spike_loss_ratio,
                self.train_args.log_grad_diff_for_debug,
                self.optimizer,
            )

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(self.module[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.
        timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
        if spike_loss_ratio == 0.0:
            logger.info(f"[Rank {torch.distributed.get_rank()}] spike loss ratio is 0, skip optimizer.step()")
            update_successful = False
            grad_norm = get_grad_norm_in_optimizer(self.optimizer) if total_grad_norm is None else total_grad_norm
            num_zeros_in_grad = count_zeros_in_grad(self.optimizer) if args.log_num_zeros_in_grad else None
        else:
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers("optimizer").stop()

        if is_megatron_version_bigger_than("0.11.0"):
            from megatron.training.utils import (
                logical_and_across_model_parallel_group,
                reduce_max_stat_across_model_parallel_group,
            )

            # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
            # so we must gather across mp ranks
            update_successful = logical_and_across_model_parallel_group(update_successful)
            # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
            # so we must gather across mp ranks
            grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
            if args.log_num_zeros_in_grad:
                num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(num_zeros_in_grad)

        # Vision momentum.
        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(self.module[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
            self.scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        return loss_to_log, skipped_iter, grad_norm, num_zeros_in_grad

    def eval_step(self, data_iterator):
        """
        Evaluation step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to evaluate on.
        """

        args = get_args()

        # make validation batch size independent from training batch size
        eval_batch_size = args.global_batch_size
        eval_num_microbatches = eval_batch_size // (args.micro_batch_size * args.data_parallel_size)

        forward_backward_func = get_forward_backward_func()
        # Don't care about timing during evaluation
        self.module_config.timers = None
        # loss_dicts: A list of loss whose length equals to the number of microbatches.
        loss_dicts = forward_backward_func(
            forward_step_func=self.train_step_handler.get_forward_step_func(),
            data_iterator=data_iterator,
            model=self.module,
            num_microbatches=eval_num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=True,
        )
        self.module_config.timers = get_timers()

        # Empty unused memory
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        monitored_dict = {"losses_reduced": loss_dicts}

        try:
            loss_postprocessed = self.train_step_handler.loss_postprocessing(monitored_dict)

            assert (
                "loss_to_log" in loss_postprocessed
            ), f"'loss_to_log' should be in return value of {type(self.train_step_handler)}::loss_postprocessing()."

            loss_to_log = loss_postprocessed["loss_to_log"]
        except Exception as e:
            # Compat old loss postprocessing method
            if self.train_args.is_last_process:
                logger.warning(f" {type(e).__name__} {e}")
                logger.warning(
                    f"Can't get eval loss from {type(self.train_step_handler)}::loss_postprocessing(),"
                    f" please upgrade your {type(self.train_step_handler)}::loss_postprocessing() function!"
                )
            loss_to_log = self.train_step_handler.validation_loss_postprocessing(loss_dicts)

        args.consumed_valid_samples += eval_batch_size

        return loss_to_log

    def log_eval_results(self):
        if self.iteration == 0 or len(self.eval_total_loss_dict) == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f"validation/test loss at iteration {self.iteration} | "
        for key, value in self.eval_total_loss_dict.items():
            if key.endswith("_num_iters") or torch.numel(value) > 1:
                continue
            value = value / self.eval_total_loss_dict[key + "_num_iters"]
            string += f"{key} value: {value} | "
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f"{key} PPL: {ppl} | "
            if writer:
                if not key.endswith("validation") and not key.endswith("test"):
                    key = key + " validation"
                writer.add_scalar(f"{key}", value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f"{key} ppl", ppl, self.iteration)

        length = len(string) + 1
        print_rank_last("-" * min(length, 150))
        print_rank_last(string)
        print_rank_last("-" * min(length, 150))
        self.eval_total_loss_dict = {}

    def training_log(self, **kwargs) -> Dict:
        if self.training_log_args is not None:
            self.report_memory_flag, logging_metrics = training_log(*self.training_log_args)
            # To ensure only once call in a step, set self.training_log_args to None
            self.training_log_args = None
            return logging_metrics
        return {}

    def save_checkpoint(
        self,
        output_dir: Optional[Path] = None,
        best_model_checkpoint=None,
        **kwargs,
    ):
        def _save_trainer_state():
            if torch.distributed.get_rank() == 0:
                try:
                    checkpoint_dir_path = self.get_checkpoint_iteration_path_dir(None, return_base_dir=True)
                    trainer_state_path = str(checkpoint_dir_path.joinpath(TRAINER_STATE_NAME))
                    self.train_state.save_to_json(trainer_state_path)
                    logger.info(f"Successfully save trainer_state.json at {trainer_state_path}")
                except Exception as e:
                    logger.error(f"Fail to save {trainer_state_path}! {e}")

        custom_async_finalize_fn = None
        if "custom_async_finalize_fn" in inspect.signature(save_checkpoint).parameters:
            custom_async_finalize_fn = _save_trainer_state
            kwargs.update(custom_async_finalize_fn=custom_async_finalize_fn)

        timers = get_timers()
        timers("save-checkpoint", log_level=0).start(barrier=True)

        if should_disable_forward_pre_hook():
            self.disable_forward_pre_hook()
        self.ckpt_saver.save(
            iteration=self.iteration,
            output_dir=str(output_dir),
            train_args=self.train_args,
            module=self.module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            num_floating_point_operations_so_far=self.num_floating_point_operations_so_far,
            best_model_checkpoint=best_model_checkpoint,
            train_state=self.train_state,
            **kwargs,
        )
        if should_disable_forward_pre_hook():
            self.enable_forward_pre_hook()

        timers("save-checkpoint").stop(barrier=True)
        timers.log(["save-checkpoint"])

        gc.collect()

        if custom_async_finalize_fn is None:
            _save_trainer_state()

    def get_checkpoint_iteration_path_dir(self, output_dir: Optional[Path], **kwargs) -> Path:
        """
        this function will return the real ckpt save iteration folder path, and make sure the main process will
        generate the folder.
        Args:
            output_dir: the output_dir from the config. If None, will get from megatron args().save
            **kwargs:

        Returns:
            the real ckpt save iteration folder path

        """
        args = get_args()
        output_dir = output_dir or Path(args.save)
        kwargs["return_base_dir"] = True
        checkpoint_name = self.ckpt_saver.get_interation_path(
            output_dir=str(output_dir), iteration=self.iteration, **kwargs
        )
        if self.train_args.is_main_process:
            os.makedirs(checkpoint_name, exist_ok=True)

        return Path(checkpoint_name)

    def load_checkpoint(self, resume_from_ckpt: Optional[Path], model, optimizer=None, scheduler=None, **kwargs):
        args = get_args()

        if resume_from_ckpt is not None:
            args.load = str(resume_from_ckpt)

        iteration, num_floating_point_operations_so_far = self.ckpt_loader.load(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_args=self.train_args,
            **kwargs,
        )

        self.iteration = iteration

        if iteration > 0:  # not finetune, continue pretrain
            trainer_state_path = self.get_checkpoint_iteration_path_dir(output_dir=get_args().load)
            AtorchTrainerState.load_from_json(str(trainer_state_path.joinpath(TRAINER_STATE_NAME)), self.train_state)

        return iteration, num_floating_point_operations_so_far

    def optimizer_zero_grad(self):
        # Set grad to zero.
        for model_chunk in self.module:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

    def disable_forward_pre_hook(self, param_sync=True, pre_hook_enabled=None):
        if is_megatron_version_bigger_than("0.10.0"):
            from megatron.training.training import disable_forward_pre_hook

            if "param_sync" in inspect.signature(disable_forward_pre_hook).parameters:
                disable_forward_pre_hook(self.module, param_sync=param_sync)
            else:
                disable_forward_pre_hook(self.module)
            if pre_hook_enabled is not None:
                self.pre_hook_enabled = pre_hook_enabled
        else:
            self.optimizer.disable_pre_hook()

    def enable_forward_pre_hook(self, pre_hook_enabled=None):
        if is_megatron_version_bigger_than("0.10.0"):
            from megatron.training.training import enable_forward_pre_hook

            enable_forward_pre_hook(self.module)
            if pre_hook_enabled is not None:
                self.pre_hook_enabled = pre_hook_enabled
        else:
            self.optimizer.enable_pre_hook()


class MegatronCallback(AtorchTrainerCallback):
    """
    A [`AtorchTrainerCallback`] that supplements Megatron training process.
    """

    def __init__(self, train_engine: AtorchMegatronEngine):
        super().__init__()
        self.train_engine = train_engine
        self.eval_type = None

    def on_train_begin(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of training, after Megatron initialization and building dataloader.
        """
        module = self.train_engine.module
        optimizer = self.train_engine.optimizer

        args = get_args()
        config = get_model_config(module[0])
        # Setup some training config params
        config.grad_scale_func = optimizer.scale_loss
        config.timers = get_timers()
        if isinstance(module[0], MegatronDDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            config.no_sync_func = [model_chunk.no_sync for model_chunk in module]
            if len(module) == 1:
                config.no_sync_func = config.no_sync_func[0]
            should_delay_grad_reduce = (
                args.delay_grad_reduce if hasattr(args, "delay_grad_reduce") else args.align_grad_reduce
            )
            if should_delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in module]
                if len(module) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        should_delay_param_gather = (
            args.delay_param_gather if hasattr(args, "delay_param_gather") else args.align_param_gather
        )
        if args.overlap_param_gather and should_delay_param_gather:
            if hasattr(optimizer, "finish_param_sync"):
                config.param_sync_func = [
                    lambda x: optimizer.finish_param_sync(model_index, x) for model_index in range(len(module))
                ]
            else:
                config.param_sync_func = [model_chunk.start_param_sync for model_chunk in module]
            if len(module) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads

        # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
        # or random initialization don't propagate to all ranks in first all-gather (which is a
        # no-op if things work correctly).
        if should_disable_forward_pre_hook():
            """Block forward pre-hook for certain configurations."""
            self.train_engine.disable_forward_pre_hook(param_sync=False, pre_hook_enabled=False)
            # Also remove param_sync_func temporarily so that sync calls made in
            # `forward_backward_func` are no-ops.
            self.train_engine.param_sync_func = config.param_sync_func
            config.param_sync_func = None

        # print model config
        if atorch_training_args.is_local_main_process and atorch_training_args.debug_switch.get("print_config", True):
            logger.info(f"************************ {config.__class__.__name__} ************************")
            logger.info(f"model config: {config}")
            logger.info(f"************************ {config.__class__.__name__} ************************")

        self.train_engine.module_config = config

        # =============================================
        # moe_routing_map load and save
        # =============================================
        # Save moelayer routing map
        if getattr(args, "moe_router_save", False):
            # if save routing map dir and iters must be specified
            if (
                args.moe_router_save_dir is None
                or args.moe_router_save_iters is None
                or args.moe_splits_save_dir is None
            ):
                raise ValueError(
                    "When --moe-router-save is True, "
                    "--moe-router-save-dir, --moe-router-save-iters "
                    "and --moe-splits-save_dir must be specified"
                )
            if args.moe_token_dispatcher_type != "alltoall":
                raise ValueError("--moe-router-save only supports --moe-token-dispatcher-type=alltoall")
            # mkdir if not exists
            os.makedirs(args.moe_router_save_dir, exist_ok=True)
            os.makedirs(args.moe_splits_save_dir, exist_ok=True)
            # save iters
            str_list = args.moe_router_save_iters.split(",")
            int_list = [int(x) for x in str_list]
            self.train_engine.saveIter = int_list

        # TODO: parse related args and check
        # load moelayer routing map
        if getattr(args, "moe_router_load", False):
            if args.moe_router_load_dir is None:
                raise ValueError("When --moe-router-load is True, --moe-router-load-dir must be specified")
        # find all the moe-layers
        if getattr(args, "moe_router_save", False) or getattr(args, "moe_router_load", False):
            MoELayerDict = {}

            def find_layers_with_path(module, target_class=MoELayer, prefix=""):
                # check if the module is target
                if isinstance(module, target_class):
                    yield (prefix, module)

                # sub_modules
                for name, child in module.named_children():
                    current_path = f"{prefix}.{name}" if prefix else name
                    yield from find_layers_with_path(child, target_class, current_path)

            for path, layer in find_layers_with_path(module[0]):
                key = f"{path.replace('.', '_')}_layerid_{layer.layer_number}"
                MoELayerDict[key] = layer
            self.train_engine.MoELayerDict = MoELayerDict

        # load moe-routing-map
        if getattr(args, "moe_router_load", False):

            def parse_routing_map_filename(file_path):
                """parse file name"""
                dir_name = os.path.basename(os.path.dirname(file_path))  # iter_100
                file_name = os.path.basename(file_path)  # layer_encoder_layer_id_2_dp_rank_0_routing_map.pt
                # parse iteration
                try:
                    iteration = int(dir_name.replace("iter_", ""))
                except ValueError:
                    raise ValueError(f"Invalid iteration directory name: {dir_name}")
                # parse other information
                pattern = (
                    r"layer_(?P<layer_name>\w+)_layer_id_(?P<layer_id>\d+)_dp_rank_(?P<dp_rank>\d+)_routing_map\.pt"
                )
                match = re.search(pattern, file_name)
                if not match:
                    logger.error(
                        f"Invalid filename format: '{file_name}'\n"
                        "Expected format: 'layer_<layer_name>_layer_id_<layer_id>_dp_rank_<dp_rank>_routing_map.pt'"
                    )
                    raise ValueError(f"Invalid filename format: {file_name}")
                return {
                    "layer_name": match.group("layer_name"),
                    "layer_id": int(match.group("layer_id")),
                    "dp_rank": int(match.group("dp_rank")),
                    "iteration": iteration,
                    "file_path": file_path,
                }

            def process_iteration_directory(iter_dir, layers_dict):
                if not os.path.isdir(iter_dir):
                    raise ValueError(f"Directory does not exist: {iter_dir}")
                # acquire all .pt files
                pt_files = [os.path.join(iter_dir, f) for f in os.listdir(iter_dir) if f.endswith(".pt")]
                for file_path in pt_files:
                    params = parse_routing_map_filename(file_path)
                    layer = MoELayerDict.get(f'{params["layer_name"]}')
                    if layer is None:
                        continue
                    layer.token_dispatcher.load_routing_map(file_path, params["dp_rank"])
                return

            # load ckpts
            process_iteration_directory(args.moe_router_load_dir, MoELayerDict)
            for layer in MoELayerDict.values():
                assert layer.token_dispatcher.preload_routing_map is not None, "Routing map not loaded"

        ###############################################

    def on_step_end(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of a training step, after forward+backward+optimizer.step,
        but before logging/evaluate/save_checkpoint
        """
        args = get_args()
        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if args.curr_iteration == self.train_engine.start_iteration:
            if self.train_engine.skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                self.train_engine.start_iteration = args.curr_iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_disable_forward_pre_hook():
                    self.train_engine.enable_forward_pre_hook(pre_hook_enabled=True)
                    self.train_engine.module_config.param_sync_func = self.train_engine.param_sync_func  # type: ignore[attr-defined] # noqa E501

    def on_evaluate_begin(  # type: ignore[override]
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of evaluation.
        """
        # "eval" or "test"
        self.eval_type = kwargs["eval_type"] if "eval_type" in kwargs and kwargs["eval_type"] is not None else "eval"

        if should_disable_forward_pre_hook():
            self.train_engine.disable_forward_pre_hook(pre_hook_enabled=False)

        args = get_args()
        timers = get_timers()
        timers(f"{self.eval_type}-time", log_level=0).start(barrier=True)

        if args.vision_pretraining and args.vision_pretraining_type == "dino":
            from megatron.legacy.model.vision.knn_monitor import compute_feature_bank

            compute_feature_bank(self.train_engine.module)

    def on_evaluate(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called after evaluation.
        """
        # Print eval result
        self.train_engine.log_eval_results()

        timers = get_timers()
        timers(f"{self.eval_type}-time").stop()
        timers.log([f"{self.eval_type}-time"])

        if should_disable_forward_pre_hook():
            self.train_engine.enable_forward_pre_hook(pre_hook_enabled=True)

    def on_save_begin(  # type: ignore[override]
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of saving.
        """
        pass

    def on_save(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_train_end(
        self,
        atorch_training_args: AtorchTrainingArgs,
        state: AtorchTrainerState,
        control: AtorchTrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.
        """
        # Flush TensorBoard, WandB writers and one-logger.
        writer = get_tensorboard_writer()
        if writer:
            writer.flush()

        # Close out pre-hooks if using distributed optimizer and overlapped param gather.
        if self.train_engine.pre_hook_enabled:
            self.train_engine.disable_forward_pre_hook()
