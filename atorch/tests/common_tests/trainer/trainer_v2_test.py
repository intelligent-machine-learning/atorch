import glob
import json
import os
import sys
from functools import partial
from typing import Union
from unittest.mock import MagicMock, patch
from urllib.request import urlretrieve

import pytest
import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.log_utils import default_logger as logger

pytestmark = pytest.mark.core24
pytest.importorskip("torch", minversion="2.0.9")

python_version = sys.version_info

from atorch.common.util_func import find_free_port  # noqa: E402
from atorch.trainer.args import AtorchTrainingArgs  # noqa: E402
from atorch.trainer.atorch_trainer_v2 import AtorchTrainerV2  # noqa: E402
from atorch.trainer.megatron import MegatronTrainStep  # noqa: E402
from atorch.trainer.utils import DistributedType  # noqa: E402
from atorch.utils.import_util import is_megatron_lm_available  # noqa: E402
from atorch.utils.version import is_megatron_version_bigger_than, torch_version  # noqa: E402

assert is_megatron_lm_available(), f"Can't import megatron, PYTHONPATH={os.environ['PYTHONPATH']}"

if is_megatron_lm_available():
    import megatron.legacy.model
    from megatron.core import mpu
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import import_module
    from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
    from megatron.training import get_args, get_tokenizer, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.utils import (
        average_losses_across_data_parallel_group,
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml


def model_provider(pre_process=True, post_process=True) -> Union["GPTModel", "megatron.legacy.model.GPTModel"]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
    else:
        assert args.context_parallel_size == 1, "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config, num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process
        )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    def is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    def core_gpt_dataset_config_from_args(args):
        tokenizer = get_tokenizer()

        if is_megatron_version_bigger_than("0.6.0", check_equality=False):
            from megatron.core.datasets.utils import get_blend_from_list

            return GPTDatasetConfig(
                random_seed=args.seed,
                sequence_length=args.seq_length,
                blend=get_blend_from_list(args.data_path),
                blend_per_split=[
                    get_blend_from_list(args.train_data_path),
                    get_blend_from_list(args.valid_data_path),
                    get_blend_from_list(args.test_data_path),
                ],
                # renormalize_blend_weights=args.renormalize_blend_weights,
                split=args.split,
                num_dataset_builder_threads=args.num_dataset_builder_threads,
                path_to_cache=args.data_cache_path,
                mmap_bin_files=args.mmap_bin_files,
                tokenizer=tokenizer,
                reset_position_ids=args.reset_position_ids,
                reset_attention_mask=args.reset_attention_mask,
                eod_mask_loss=args.eod_mask_loss,
                create_attention_mask=args.create_attention_mask_in_dataloader,
                s3_cache_path=args.s3_cache_path,
            )
        else:
            return GPTDatasetConfig(
                random_seed=args.seed,
                sequence_length=args.seq_length,
                blend=args.data_path,
                blend_per_split=[
                    args.train_data_path,
                    args.valid_data_path,
                    args.test_data_path,
                ],
                split=args.split,
                path_to_cache=args.data_cache_path,
                mock=args.mock_data,
                mmap_bin_files=args.mmap_bin_files,
                tokenizer=tokenizer,
                reset_position_ids=args.reset_position_ids,
                reset_attention_mask=args.reset_attention_mask,
                eod_mask_loss=args.eod_mask_loss,
                create_attention_mask=args.create_attention_mask_in_dataloader,
            )

    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


class DpoSampler(MegatronPretrainingSampler):
    def set_epoch(self, epoch):
        self.epoch = epoch


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    def get_train_valid_test_num_samples():
        """Train/valid/test num samples."""

        args = get_args()

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        if hasattr(args, "test_iters"):
            test_iters = args.test_iters
        else:
            test_iters = args.eval_iters

        return (
            train_samples,
            eval_iters * args.global_batch_size,
            test_iters * args.global_batch_size,
        )

    def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
        """Build pretraining datasets."""
        train_valid_test_num_samples = get_train_valid_test_num_samples()
        print_rank_0(" > datasets target sizes (minimum size):")
        print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
        print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
        print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
        return build_train_valid_test_datasets_provider(train_valid_test_num_samples)

    def build_pretraining_data_loader(dataset, consumed_samples):
        """Build dataloader given an input dataset."""

        if dataset is None:
            return None
        args = get_args()

        # Megatron sampler
        if args.dataloader_type == "single":
            # batch_sampler = DpoSampler(
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
        elif args.dataloader_type == "cyclic":
            batch_sampler = MegatronPretrainingRandomSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                data_sharding=args.data_sharding,
            )
        elif args.dataloader_type == "external":
            # External dataloaders are passed through. User is expected to provide a
            # torch-compatible dataloader and define samplers, if needed.
            return dataset
        else:
            raise Exception("{} dataloader type is not supported.".format(args.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
        """Build pretraining data loaders."""

        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0("> building train, validation, and test datasets ...")

        # For DPO ut
        if not hasattr(args, "iteration"):
            args.iteration = 0
            args.consumed_train_samples = 0

        # Backward compatibility, assume fixed batch size.
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, "only backward compatiblity support for iteration-based training"
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (
                    (args.iteration // args.eval_interval) * args.eval_iters * args.global_batch_size
                )

        # Rely on distributed-aware core datasets, temporary
        is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

        # Construct the data pipeline
        if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

            # Build datasets.
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(build_train_valid_test_datasets_provider)
            # Build dataloders.
            train_dataloader = build_pretraining_data_loader(train_ds, args.consumed_train_samples)
            if args.skip_train:
                valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
            else:
                valid_dataloader = build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

            # Flags to know if we need to do training/validation/testing.
            do_train = train_dataloader is not None and args.train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
            flags = torch.tensor(
                [int(do_train), int(do_valid), int(do_test)],
                dtype=torch.long,
                device="cuda",
            )
        else:
            flags = torch.tensor([0, 0, 0], dtype=torch.long, device="cuda")

        torch.distributed.broadcast(flags, 0)

        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()

        return train_dataloader, valid_dataloader, test_dataloader

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider
    )

    if train_dataloader is not None:
        logger.info(f"[Rank {args.rank}] build dataloader over!")
        logger.info(
            f"[Rank {args.rank}] train_dataloader {len(train_dataloader)} valid_dataloader {len(valid_dataloader)}"
            f" test_dataloader {len(test_dataloader)}",
        )
        logger.info(
            f"[Rank {args.rank}] train_dataset {len(train_dataloader.dataset)}"
            f" valid_dataset {len(valid_dataloader.dataset)} test_dataset {len(test_dataloader.dataset)}",
        )

    return train_dataloader, valid_dataloader, test_dataloader


class GPTTrainStep(MegatronTrainStep):
    """
    GPT train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Generate a batch."""

            # TODO: this is pretty hacky, find a better way
            if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
                return None, None, None, None, None

            # get batches based on the TP rank you are on
            batch = get_batch_on_this_tp_rank(data_iterator)

            # slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)

            return batch.values()

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask, output_tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses
            """
            args = get_args()

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            if args.context_parallel_size > 1:
                loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
                loss = loss[0] / loss[1]
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Check individual rank losses are not NaN prior to DP all-reduce.
            if args.check_for_nan_in_loss_and_grad:
                global_rank = torch.distributed.get_rank()
                assert not loss.isnan(), (
                    f"Rank {global_rank}: found NaN in local forward loss calculation. "
                    f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
                )

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        def forward_step(data_iterator, model: GPTModel):
            """Forward training step.

            Args:
                data_iterator : Input data iterator
                model (GPTModel): The GPT Model
            """
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch_func()(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask)

        return forward_step


vocab_url = "https://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/jinshi/atorch_unittest_data/vocab.json"
merge_url = "https://alps-common.oss-cn-hangzhou-zmf.aliyuncs.com/users/jinshi/atorch_unittest_data/merges.txt"
vocab_file = "/tmp/gpt2_vocab.json"
merge_file = "/tmp/gpt2_merges.json"


def download_tokenizer_file():
    try:
        if not os.path.exists(vocab_file):
            logger.info(f"Downloading {vocab_url} to {vocab_file}")
            urlretrieve(vocab_url, vocab_file)
        if not os.path.exists(merge_file):
            logger.info(f"Downloading {merge_url} to {merge_file}")
            urlretrieve(merge_url, merge_file)
    except Exception as e:
        logger.exception(f"Download {vocab_url} and {merge_url} failed, please check if the addresses exist. {e}")
        return False
    return True


def run_atorch_trainer_v2(rank):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)

    output_dir = "/tmp/output_atorch_trainer"

    # test nv dynamic profiler
    with open("/tmp/profile_config.json", "w") as f:
        json.dump(
            {
                "output_dir": "/tmp/profile",
                "start_step": 20,
                "schedule_warmup": 2,
                "schedule_active": 1,
                "with_stack": False,
                "with_flops": False,
                "with_modules": False,
                "record_shapes": False,
                "profile_memory": False,
                "acc_events": False,
                "activities": ["CPU", "CUDA"],
                "profile_ranks": [-1],
                "use_gzip": True,
            },
            f,
        )

    # test dynamic saving checkpoint
    with open("/tmp/dynamic_save_config.json", "w") as f:
        json.dump(
            {"save_at_dynamic_steps": [100]},
            f,
        )

    training_args = AtorchTrainingArgs(
        distributed_type="megatron",
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        bf16=True,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        dynamic_save_config_path="/tmp/dynamic_save_config.json",
        evaluation_strategy="steps",
        eval_steps=25,
        test_strategy="steps",
        test_steps=25,
        test_on_save=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_nan_inf_filter=False,
        gradient_checkpointing=False,
        tensorboard_dir=os.path.join(output_dir, "runs"),
        use_deterministic_algorithms=True,
        profiler_type="nv_dp",
        dynamic_profiler_config_path="/tmp/profile_config.json",
        memory_snapshot_path="/tmp/memory_snapshot",
        # finetune_type="dpo",  # to be removed
        # max_steps=25,
    )

    train_valid_test_datasets_provider.is_distributed = True

    megatron_args = dict(
        # Custom function
        custom_model_provider_function=model_provider,
        custom_megatron_dataloaders_provider_function=partial(
            build_train_valid_test_data_iterators, train_valid_test_datasets_provider
        ),
        custom_train_step_class=GPTTrainStep,
        # model args
        model_type_name="gpt",
        num_layers=16,
        hidden_size=768,
        num_attention_heads=12,
        group_query_attention=True,
        num_query_groups=12,
        max_position_embeddings=512,
        position_embedding_type="rope",
        make_vocab_size_divisible_by=1,
        norm_epsilon=1e-5,
        normalization="RMSNorm",
        untie_embeddings_and_output_weights=True,
        use_flash_attn=True,
        # tokenizer
        tokenizer_type="GPT2BPETokenizer",
        vocab_file=vocab_file,
        merge_file=merge_file,
        # optimizer
        optimizer="adam",
        # Regular args
        attention_dropout=0.0,
        hidden_dropout=0.0,
        weight_decay=1e-1,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        # Megatron training args
        pretraining_flag=True,
        use_mcore_models=True,
        transformer_impl="transformer_engine",
        micro_batch_size=1,
        global_batch_size=2,
        add_bias_linear=False,
        bias_gelu_fusion=False,
        recompute_activations=True,
        recompute_granularity="selective",
        train_iters=25,
        eval_iters=5,
        test_iters=5,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        # Distributed args
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        num_virtual_stages_per_pipeline_rank=2,
        sequence_parallel=True,
        distributed_backend="nccl",
        use_distributed_optimizer=True,
        # Logging args
        enable_one_logger=False,
        log_timers_to_tensorboard=True,
        log_validation_ppl_to_tensorboard=True,
        log_memory_to_tensorboard=True,
        log_throughput=True,
        log_params_norm=True,
        log_params_std=True,
        tensorboard_dir=training_args.tensorboard_dir,
        # Initialization args
        seed=1403,
        init_method_std=0.02,
        # Learning rate args
        lr=3e-5,
        min_lr=3e-6,
        lr_ecay_style="cosine",
        lr_warmup_fraction=0.1,
        # Data
        data_cache_path=os.path.join(training_args.output_dir, "data_cache"),
        mock_data=True,
        seq_length=512,
        num_workers=0,
        mtp_num_layers=1,
        # routing_map save
        moe_router_save=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_save_dir="/tmp/trysave",
        moe_splits_save_dir="/tmp/splitsave/",
        moe_router_save_iters="2,5,10,15",
        moe_router_load=True,
        moe_router_load_dir="/tmp/trysave/iter_10/",
    )

    if megatron_args["sequence_parallel"]:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    training_args.extra_configs = megatron_args
    example_data = {"routing_map": torch.randn(10, 10), "metadata": {"dp_rank": 1, "layer_id": 0}}
    os.makedirs("/tmp/trysave/iter_10/", exist_ok=True)
    filename = "layer_expert_layer_id_0_dp_rank_1_routing_map.pt"
    torch.save(example_data, os.path.join("/tmp/trysave/iter_10/", filename))
    trainer = AtorchTrainerV2(
        args=training_args,
    )
    train_result = trainer.train()
    print(f"{train_result.metrics}")

    atorch.reset_distributed()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
@pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")  # type: ignore
@pytest.mark.skipif(
    not (python_version.major >= 3 and python_version.minor >= 10), reason="Megatron 0.11 requires python >= 3.10"
)
@pytest.mark.parametrize("world_size", [4])
def test_atorch_trainer(world_size):

    if not download_tokenizer_file():
        logger.warning(f"Can't download {vocab_url} and {merge_url}, skip this unit test.")
        return

    # Test for AntMonitor
    if os.environ.get("ANTMONITOR_TFEVENT_PATH") is None:
        os.environ["ANTMONITOR_TFEVENT_PATH"] = "/home/admin/logs/tfevent"

    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())

    mp.spawn(
        run_atorch_trainer_v2,
        nprocs=world_size,
        join=True,
        daemon=False,
        start_method="spawn",
    )

    os.environ["MASTER_ADDR"] = ""
    os.environ["MASTER_PORT"] = ""

    # assert the gpu_num profile file is generated in /tmp/profile
    profile_files = glob.glob("/tmp/profile/*.json.gz")
    assert len(profile_files) == world_size

    # assert the profile file is not empty
    for profile_file in profile_files:
        assert os.path.getsize(profile_file) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
@pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")  # type: ignore
@pytest.mark.skipif(
    not (python_version.major >= 3 and python_version.minor >= 10), reason="Megatron 0.11 requires python >= 3.10"
)
@pytest.mark.parametrize("distributed_type", ["megatron"])
def test_evaluate_with_mocks(distributed_type):
    with patch("atorch.trainer.args.AtorchAcceleratorState") as mock_accel_state, patch(
        "atorch.trainer.atorch_trainer_v2.get_timers"
    ) as mock_get_timers, patch("atorch.trainer.atorch_trainer_v2.get_args") as mock_get_args:

        # Patch AtorchAcceleratorState to have required attributes
        mock_accel_state.return_value.distributed_type = DistributedType.MEGATRON
        mock_accel_state.return_value.is_main_process = True
        mock_accel_state.return_value.is_local_main_process = True
        mock_accel_state.return_value.local_process_index = 0

        # Patch timers
        mock_timer = MagicMock()
        mock_get_timers.return_value = MagicMock(return_value=mock_timer)
        mock_timer.start.return_value = None
        mock_timer.stop.return_value = None
        mock_timer.log.return_value = None

        # Patch megatron args
        mock_args = MagicMock()
        mock_args.test_iters = 0
        mock_args.global_batch_size = 1
        mock_get_args.return_value = mock_args

        # Import after patching AtorchAcceleratorState
        from atorch.trainer.args import AtorchTrainingArgs
        from atorch.trainer.atorch_trainer_v2 import AtorchTrainerV2

        training_args = AtorchTrainingArgs(
            distributed_type=distributed_type,
            output_dir="/tmp/test_eval",
            overwrite_output_dir=True,
            per_device_eval_batch_size=1,
            do_eval=True,
        )

        # Patch train_engine and its methods
        trainer = AtorchTrainerV2(args=training_args)

        # Patch train_engine and its methods
        trainer.train_engine = MagicMock()
        trainer.train_engine.get_dataloader.return_value = [torch.tensor(1.0), torch.tensor(2.0)]
        trainer.train_engine.eval.return_value = None
        trainer.train_engine.train.return_value = None
        trainer.train_engine.train_step_handler.model_output_class = None

        # Patch callback_handler
        trainer.callback_handler = MagicMock()
        trainer.callback_handler.on_evaluate_begin.return_value = None
        trainer.callback_handler.on_prediction_step.return_value = None
        trainer.callback_handler.on_evaluate.return_value = None

        # Patch log
        trainer.log = MagicMock()

        # Patch control/state
        trainer.control = MagicMock()
        trainer.state = MagicMock()

        # Run evaluate
        result = trainer.evaluate(eval_or_test="test")
        print(result)
        assert isinstance(result, dict)
        assert "test_loss" in result or "test_perplexity" in result

        trainer.train_engine.get_dataloader.return_value = None
        result = trainer.evaluate(eval_or_test="test")
        print(result)
        assert result is None
