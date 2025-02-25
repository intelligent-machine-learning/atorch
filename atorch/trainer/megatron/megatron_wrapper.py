"""
Megatron wrapper.
"""
import dataclasses
import math
import os
import random
import sys
from typing import Dict, List, Union

import numpy as np
import torch
import torch.distributed
from torch.utils.data import DataLoader

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.atorch_train_engine import AtorchTrainEngine
from atorch.trainer.megatron.megatron_dataloader import (
    AtorchMegatronDataloader,
    _prepare_megaton_dataloader,
    wrap_megatron_dataloader,
)
from atorch.trainer.megatron.megatron_train_step import BertTrainStep, GPTTrainStep, MegatronTrainStep, T5TrainStep
from atorch.trainer.utils import (
    broadcast_spike_loss_ratio_in_pp_group,
    calc_params_std,
    is_main_process,
    scale_main_grad_for_spike_loss,
    training_log,
)
from atorch.utils.import_util import is_megatron_lm_available, is_torch_npu_available

if is_megatron_lm_available():
    from megatron.core import mpu, tensor_parallel
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.utils import get_model_config
    from megatron.legacy.model import BertModel, GPTModel, T5Model
    from megatron.legacy.model.classification import Classification
    from megatron.legacy.model.module import MegatronModule
    from megatron.training import get_args, get_tensorboard_writer, print_rank_last

    try:
        from megatron.training import get_num_microbatches, update_num_microbatches
    except ImportError:
        from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
    from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
    from megatron.training.checkpointing import get_checkpoint_name, load_args_from_checkpoint
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
    from megatron.training.utils import calc_params_l2_norm, print_rank_0, unwrap_model
    from megatron.training.yaml_arguments import validate_yaml


DATALOADER_INDEX_MAPPER = dict(train=0, eval=1, test=2)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    use_deterministic_algorithms=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."
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
        # TODO: extra key.
        setattr(args, key, value)

    device_count = torch.cuda.device_count()
    args.local_rank = torch.distributed.get_rank() % device_count

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        import inspect

        if "get_embedding_ranks" not in inspect.signature(_initialize_distributed).parameters:
            _initialize_distributed()
        else:
            _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

        if use_deterministic_algorithms:
            torch.use_deterministic_algorithms(True)

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


def prepare_optimizer(args, megatron_args, model):
    logger.info("Preparing optimizer")
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = get_timers()
    if hasattr(args, "use_local_sgd") and args.use_local_sgd:
        from atorch.local_sgd.configs import GTAConfig, LocalSGDConfig, OuterOptimizerConfig
        from atorch.local_sgd.megatron import get_megatron_optimizer

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
        return get_megatron_optimizer(
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
        from megatron.core.optimizer import get_megatron_optimizer

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


class AtorchMegatronEngine(AtorchTrainEngine):
    def __init__(
        self,
        train_args: AtorchTrainingArgs,
        model: MegatronModule,
        optimizer: MegatronOptimizer,
        scheduler: OptimizerParamScheduler,
        dataloaders: Union[AtorchMegatronDataloader, tuple, None],
        resume_from_checkpoint: str = None,
        **kwargs,
    ):
        super().__init__(train_args)

        self.module: List[MegatronModule]
        self.optimizer: MegatronOptimizer
        self.scheduler: OptimizerParamScheduler
        self.dataloaders = dataloaders

        self._dataloaders: List[DataLoader] = []

        self.megatron_args = train_args.megatron_args()

        self.initialize(self.megatron_args, self.train_args.use_deterministic_algorithms)

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
        (
            self.module,
            self.optimizer,
            self.scheduler,
        ) = self.prepare_model_optimizer_scheduler(resume_from_checkpoint=resume_from_checkpoint)

        if self.train_args.convert_checkpoint:
            from megatron.training.checkpointing import save_checkpoint

            args.save = self.train_args.output_dir_for_converting
            args.no_save_optim = True
            args.no_save_rng = True

            # TODO: consider async saving
            assert args.ckpt_format == "torch_dist", "Only support to convert torch bin to torch_dist, so you must set "
            assert args.iteration == 0, "Only support finetune from 0th-step checkpoint."

            save_checkpoint(
                args.iteration,
                self.module,
                None,
                None,
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

        # TODO: define a function to unify barrier operator.
        torch.distributed.barrier()

        # args.iteration = 0
        # args.num_floating_point_operations_so_far = 0
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
        self.num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
        self.module_config = None
        self.training_log_args = None
        self.custom_training_log_dict = None
        self.num_microbatches = get_num_microbatches()

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

        write_args_to_tensorboard()

    def _validate_train_args(
        self,
    ):
        args = get_args()
        self.train_args.per_device_train_batch_size = args.micro_batch_size * get_num_microbatches()
        self.train_args.per_device_eval_batch_size = args.micro_batch_size * get_num_microbatches()

        if args.profile:
            if is_torch_npu_available():
                logger.warning("nsys profiling is not supported on NPU env.")
            else:
                if self.train_args.profiler_type is None:
                    self.train_args.profiler_type = "nsys"
                elif self.train_args.profiler_type != "nsys":
                    raise ValueError(
                        f"Can't use {self.train_args.profiler_type} and nsys to profile "
                        "simultaneously. You can set atorch trainer's 'profiler_type' arg "
                        "to 'nsys' or not set Megatron's 'profile' arg."
                    )

        def _assert(condition, arg_name):
            assert condition, f"{arg_name} is not supported in AtorchTrainer."

        _assert(not args.enable_one_logger, "enable_one_logger")
        _assert(not args.adlr_autoresume, "adlr_autoresume")
        _assert(not args.exit_signal_handler, "exit_signal_handler")
        _assert(args.exit_duration_in_mins is None, "exit_duration_in_mins")
        _assert(args.exit_interval is None, "exit_interval")
        _assert(not args.vision_pretraining, "vision_pretraining")

    @staticmethod
    def initialize(megatron_args, use_deterministic_algorithms=False):  # todo type  type: ignore[override]
        initialize_megatron(
            extra_args_provider=megatron_args.extra_args_provider,
            args_defaults=megatron_args.to_dict(),
            ignore_unknown_args=True,
            use_deterministic_algorithms=use_deterministic_algorithms,
        )

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        if use_deterministic_algorithms:
            TORCH_MAJOR = int(torch.__version__.split(".")[0])
            TORCH_MINOR = int(torch.__version__.split(".")[1])
            if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
                torch._C._jit_set_nvfuser_enabled(False)

            args = get_args()
            if not is_torch_npu_available() and args.bias_dropout_fusion:
                # On GPU env, bias_dropout_fusion will effect the accuracy of loss and grad when resuming
                # training from a checkpoint. So if you want to use deterministic algorithms, set it to False.
                raise ValueError("You should set 'bias_dropout_fusion' to False when using deterministic algorithm.")

    def prepare_model_optimizer_scheduler(self, resume_from_checkpoint=None):
        logger.info("Preparing model optimizer scheduler")
        args = get_args()
        megatron_args = self.megatron_args
        timers = get_timers()
        if megatron_args.custom_prepare_model_function is not None:
            if megatron_args.custom_model_provider_function is None:
                raise ValueError(
                    "You must provide a `custom_model_provider_function` when using a `custom_prepare_model_function`."
                )
            custom_model_provider_func = megatron_args.custom_model_provider_function
            model = megatron_args.custom_prepare_model_function(custom_model_provider_func)
        else:
            model_type = ModelType.encoder_or_decoder
            if args.model_type_name == "t5":
                model_type = ModelType.encoder_and_decoder
            if megatron_args.custom_model_provider_function is not None:
                model_provider_func_ = megatron_args.custom_model_provider_function
            else:
                model_provider_func_ = model_provider_func
            if hasattr(args, "use_local_sgd") and args.use_local_sgd:
                from atorch.local_sgd.megatron import get_model

                model = get_model(model_provider_func_, model_type)
            else:
                from megatron.training.training import get_model

                model = get_model(model_provider_func_, model_type)

        optimizer = prepare_optimizer(args, megatron_args, model)
        scheduler = get_optimizer_param_scheduler(optimizer)

        if resume_from_checkpoint is not None:
            timers("load-checkpoint", log_level=0).start(barrier=True)

            (args.iteration, args.num_floating_point_operations_so_far,) = self.load_checkpoint(
                input_dir=resume_from_checkpoint,
                module=model,
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

        unwrapped_model = unwrap_model(model)

        # get model without FP16 and/or DDP wrappers
        if (
            args.iteration == 0
            and len(unwrapped_model) == 1
            and hasattr(unwrapped_model[0], "init_state_dict_from_bert")
        ):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        self.iteration = args.iteration
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
        def _build_dataloader():
            if self.megatron_args.custom_megatron_dataloaders_provider_function is not None:
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

        args = get_args()
        timers = get_timers()

        timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
        if args.virtual_pipeline_model_parallel_size is not None:
            train_data_iterator = []
            valid_data_iterator = []
            test_data_iterator = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                iterators = _build_dataloader()
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

        is_finetune = self.train_args.finetune_type is not None

        self._dataloaders.append(wrap_megatron_dataloader(train_data_iterator, is_finetune))
        self._dataloaders.append(wrap_megatron_dataloader(valid_data_iterator, is_finetune))
        self._dataloaders.append(wrap_megatron_dataloader(test_data_iterator, is_finetune))

        # Calculate train_iters
        if is_finetune:
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

    # TODO(@jinshi.cl): Should be called at init, instead when AtorchMegatronEngine.train() or
    # AtorchMegatronEngine.eval()
    def get_module_config(self):
        args = get_args()
        config = get_model_config(self.module[0])
        # Setup some training config params
        config.grad_scale_func = self.optimizer.scale_loss
        config.timers = get_timers()
        if isinstance(self.module[0], MegatronDDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.module]
            if len(self.module) == 1:
                config.no_sync_func = config.no_sync_func[0]
            should_delay_grad_reduce = (
                args.delay_grad_reduce if hasattr(args, "delay_grad_reduce") else args.align_grad_reduce
            )
            if should_delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.module]
                if len(self.module) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        should_delay_param_gather = (
            args.delay_param_gather if hasattr(args, "delay_param_gather") else args.align_param_gather
        )
        if args.overlap_param_gather and should_delay_param_gather:
            if hasattr(self.optimizer, "finish_param_sync"):
                config.param_sync_func = [
                    lambda x: self.optimizer.finish_param_sync(model_index, x)
                    for model_index in range(len(self.module))
                ]
            else:
                config.param_sync_func = [model_chunk.start_param_sync for model_chunk in self.module]
            if len(self.module) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        return config

    def train(self):
        for model_module in self.module:
            model_module.train()

        if self.module_config is None:
            self.module_config = self.get_module_config()

        self.log_eval_results()

    def eval(self):
        for model_module in self.module:
            model_module.eval()

        if self.module_config is None:
            self.module_config = self.get_module_config()

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
            # Update number of microbatches first without consistency. Then run consistency check
            # to make sure training configuration is still valid.
            update_num_microbatches(args.consumed_train_samples, consistency_check=False)
            if get_num_microbatches() != self.num_microbatches and self.iteration != 0:
                assert (
                    get_num_microbatches() > self.num_microbatches
                ), "number of microbatches should be increasing due to batch size rampup"
            self.num_microbatches = get_num_microbatches()
            update_num_microbatches(args.consumed_train_samples, consistency_check=True)

            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.train_step(data_iterator)

            self.iteration += 1
            args.global_step = self.iteration
            batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            args.consumed_train_samples += batch_size
            self.num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

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
                skipped_iter,
                grad_norm,
                params_norm,
                params_std,
                num_zeros_in_grad,
                custom_training_log_dict,
            ]
        else:
            loss_dict = self.eval_step(data_iterator)
            for key, value in loss_dict.items():
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

    def get_batch_data_iterator(self, batch_data):
        args = get_args()
        num_microbatches = get_num_microbatches()
        data_chunks = []
        if len(batch_data) > 0:
            if num_microbatches > 1:
                for i in range(0, num_microbatches):
                    data_chunks.append(
                        {
                            k: v[i * args.micro_batch_size : (i + 1) * args.micro_batch_size]
                            for k, v in batch_data.items()
                        }
                    )
            else:
                data_chunks = [batch_data]

        if len(self.module) > 1:
            batch_data_iterator = (
                [iter(data_chunks) for _ in range(len(self.module))]
                if len(batch_data) > 0
                else [None] * len(self.module)
            )
        else:
            batch_data_iterator = iter(data_chunks) if len(batch_data) > 0 else None
        return batch_data_iterator

    def train_step(self, data_iterator):
        """
        Training step for Megatron-LM

        Args:
            batch_data (:obj:`dict`): The batch data to train on.
        """

        args = get_args()
        timers = get_timers()

        # Set grad to zero.
        for model_chunk in self.module:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
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

        loss_processed = self.train_step_handler.loss_postprocessing(losses_reduced)

        # Compat previous-version loss_postprocessing()
        if isinstance(loss_processed, tuple):
            assert len(loss_processed) == 2
            loss_to_log, spike_loss_ratio = loss_processed
        else:
            loss_to_log = loss_processed
            spike_loss_ratio = None

        spike_loss_ratio = broadcast_spike_loss_ratio_in_pp_group(spike_loss_ratio)

        if spike_loss_ratio is not None:
            logger.info(f"[Rank {torch.distributed.get_rank()}] apply spike loss on grad with ratio {spike_loss_ratio}")
            scale_main_grad_for_spike_loss(self.module, spike_loss_ratio, self.train_args.log_grad_diff_for_debug)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
        if spike_loss_ratio == 0.0:
            logger.info(f"[Rank {torch.distributed.get_rank()}] spike loss ratio is 0, skip optimizer.step()")
            update_successful, grad_norm, num_zeros_in_grad = (False, None, None)
        else:
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers("optimizer").stop()

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

        # batch_data_iterator = self.get_batch_data_iterator(batch_data)
        forward_backward_func = get_forward_backward_func()
        # Don't care about timing during evaluation
        self.module_config.timers = None
        loss_dicts = forward_backward_func(
            forward_step_func=self.train_step_handler.get_forward_step_func(),
            data_iterator=data_iterator,
            model=self.module,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )
        self.module_config.timers = get_timers()

        # Empty unused memory
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        args.consumed_valid_samples += (
            mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in loss_dicts[0]:
                losses_reduced_for_key = [x[key] for x in loss_dicts]
                if len(losses_reduced_for_key[0].shape) == 0:
                    loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
                else:
                    loss_reduced[key] = torch.concat(losses_reduced_for_key)
            return loss_reduced
        return {}

    def log_eval_results(self):
        args = get_args()
        if self.iteration == 0 or len(self.eval_total_loss_dict) == 0:
            return
        args = get_args()
        writer = get_tensorboard_writer()
        string = f"validation loss at iteration {self.iteration} | "
        for key, value in self.eval_total_loss_dict.items():
            if key.endswith("_num_iters") or torch.numel(value) > 1:
                continue
            value = value / self.eval_total_loss_dict[key + "_num_iters"]
            string += f"{key} value: {value} | "
            ppl = math.exp(min(20, value.item()))
            if args.pretraining_flag:
                string += f"{key} PPL: {ppl} | "
            if writer:
                writer.add_scalar(f"{key} validation", value.item(), self.iteration)
                if args.pretraining_flag:
                    writer.add_scalar(f"{key} validation ppl", ppl, self.iteration)

        length = len(string) + 1
        print_rank_last("-" * length)
        print_rank_last(string)
        print_rank_last("-" * length)
        self.eval_total_loss_dict = {}

    def training_log(self, **kwargs) -> Dict:
        if self.training_log_args is not None:
            self.report_memory_flag, logging_metrics = training_log(*self.training_log_args)
            return logging_metrics
        return {}

    def save_checkpoint(
        self,
        output_dir=None,
        trainer_state: dict = None,
        best_model_checkpoint=None,
        **kwargs,
    ):
        self.log_eval_results()
        args = get_args()

        if self.train_args.flash_checkpoint:
            from dlrover.trainer.torch.flash_checkpoint.megatron_async_save.megatron_async_torch_save import (
                save_checkpoint,
            )

            save_checkpoint(
                self.iteration,
                self.module,
                self.optimizer,
                self.scheduler,
                num_floating_point_operations_so_far=self.num_floating_point_operations_so_far,
                timeout=self.train_args.flash_checkpoint_timeout,
            )

            from atorch.trainer.base.checkpoint import _rotate_checkpoints

            ckpt_save_total_limit = (
                None if self.train_args.save_total_limit is None else self.train_args.save_total_limit + 2
            )

            if is_main_process():
                _rotate_checkpoints(
                    output_dir=output_dir,
                    save_total_limit=ckpt_save_total_limit,
                    best_model_checkpoint=best_model_checkpoint,
                )

            torch.distributed.barrier()

            # ais will handle the ckpt rotate deletion.
        else:  # save locally sync mode
            checkpoint_dir_path = self.get_checkpoint_path_dir(args.save, return_base_dir=True)

            if is_main_process():
                os.makedirs(checkpoint_dir_path, exist_ok=True)

            torch.distributed.barrier()

            from megatron.training.checkpointing import save_checkpoint

            save_checkpoint(
                self.iteration,
                self.module,
                self.optimizer,
                self.scheduler,
                num_floating_point_operations_so_far=self.num_floating_point_operations_so_far,
            )

            torch.distributed.barrier()

            from atorch.trainer.base.checkpoint import _rotate_checkpoints

            if is_main_process():
                _rotate_checkpoints(
                    output_dir=output_dir,
                    save_total_limit=self.train_args.save_total_limit,
                    best_model_checkpoint=best_model_checkpoint,
                )

        torch.distributed.barrier()

    def get_checkpoint_path_dir(self, output_dir, **kwargs):
        args = get_args()
        checkpoint_name = get_checkpoint_name(args.save, self.iteration, **kwargs)
        return checkpoint_name
        # return os.path.dirname(checkpoint_name)

    def load_checkpoint(self, input_dir, module, optimizer, scheduler, **kwargs):
        args = get_args()
        args.load = input_dir
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0

        torch.distributed.barrier()

        # we decide to use sync load for all case
        from megatron.training.checkpointing import load_checkpoint

        iteration, num_floating_point_operations_so_far = load_checkpoint(module, optimizer, scheduler)
        torch.distributed.barrier()
        return iteration, num_floating_point_operations_so_far

    def optimizer_step(self):
        # Megatron's train_step() contains optimizer.step()
        pass
        # return self.optimizer.step()

    def scheduler_step(self):
        # Megatron's train_step() contains scheduler.step()
        pass
        # return self.scheduler.step()

    def optimizer_zero_grad(self):
        return self.optimizer.zero_grad()

    def backward(self, loss):
        pass
