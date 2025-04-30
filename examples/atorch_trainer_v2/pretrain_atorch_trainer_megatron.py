#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import yaml  # type: ignore[import]
from packaging.version import Version
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser, TrainerCallback
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

try:
    import ant_patches
except ModuleNotFoundError as e:
    print(e)
    print(
        "Can't import ant_patches, if you want to use megatron with version >= 'core_r0.9.0', "
        "please use 'ant_core_r0.9.0' branch."
    )
    ant_patches = None

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.atorch_trainer_v2 import AtorchTrainerV2
from atorch.trainer.megatron import MegatronTrainStep
from atorch.utils.import_util import is_megatron_lm_available
from atorch.utils.version import get_megatron_version, is_megatron_version_bigger_than

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
    from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version(
    "datasets>=2.14.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    vocab_file: Optional[str] = field(default=None, metadata={"help": "The vocab file (a json file)."})
    merge_file: Optional[str] = field(default=None, metadata={"help": "The merge file (a json file)."})
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    tokenizer_model: Optional[str] = field(default=None, metadata={"help": "The path to tokenizer model."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the "
                "pretrained weights are loaded. set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data path (preprocessed Megatron data)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.data_path is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
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
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
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

        if is_megatron_version_bigger_than("0.10.0"):
            from megatron.training.utils import get_blend_and_blend_per_split

            # Sometimes --data-path is too long, instead we parse it from a file.
            blend: Optional[Tuple[List[str], Optional[List[float]]]]
            blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
            blend, blend_per_split = get_blend_and_blend_per_split(args)

            other_args = {}
            if get_megatron_version() == Version("0.10.0"):
                other_args.update(
                    renormalize_blend_weights=args.renormalize_blend_weights,
                )

            return GPTDatasetConfig(
                random_seed=args.seed,
                sequence_length=args.seq_length,
                blend=blend,
                blend_per_split=blend_per_split,
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
                **other_args,
            )
        elif is_megatron_version_bigger_than("0.6.0", check_equality=False):
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
                renormalize_blend_weights=args.renormalize_blend_weights,
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

        ##################################
        # Just for testing spike loss
        self.last_loss = None
        # Just for testing spike loss
        ##################################

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
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses

            Returns:
                the loss scalar for this micro-batch
                the number of non-padded tokens in this microbatch
                a dict containing reporting metrics on the loss and number of tokens across
                    the data parallel ranks
            """
            args = get_args()

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            total_tokens = loss_mask.sum()
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

            if args.context_parallel_size > 1:
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

            # Reduce loss for logging.
            reporting_loss = loss.clone().detach()
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

            local_num_tokens = loss[1].clone().detach().to(torch.int)
            return (
                loss[0] * args.context_parallel_size,
                local_num_tokens,
                {"lm loss": (reporting_loss[0], reporting_loss[1])},
            )

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

    def loss_postprocessing(self, monitor_dict):
        """
        Loss postprocessing. Average losses across all micro-batches.

        Args:
            monitor_dict: a dict to hold all related monitored matrix
                In train process: {losses_reduced: [], total_grad_norm: float or None}
                    losses_reduced: (List[torch.Tensor]):
                        A list of losses whose length equals to the number of microbatches, `global_batch_size/data_parallel_size/micro_batch_size`.  # noqa E501
                    total_grad_norm (Only in training process):
                        total grad_norm (for all params).
                In eval or test process: {losses_reduced: []}
                    losses_reduced: (List[torch.Tensor])
        Returns:
            A dict:
                In train process: return {"loss_to_log": Dict, "spike_loss_ratio": float or None}
                    The first one is a train loss dict to log, and the second one is a ratio if the spike loss occurs.
                In eval or test process: return {"loss_to_log": Dict}
        """

        args = get_args()

        assert "losses_reduced" in monitor_dict

        losses_reduced = monitor_dict.get("losses_reduced", None)

        # args.forward_mode is a internal intermediate variable to indicate current forward mode.
        # args.forward_mode belongs to ["train", "eval", "test"]
        if args.forward_mode == "train":
            total_grad_norm = monitor_dict.get("total_grad_norm", None)  # noqa: F841
            res = {"loss_to_log": {}, "spike_loss_ratio": None}
        else:
            res = {"loss_to_log": {}}

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator
            ##################################
            # Just for testing spike loss
            ratio = None
            if (
                self.last_loss is not None
                and loss_reduced["lm loss"] > self.last_loss
                and torch.abs(loss_reduced["lm loss"] - self.last_loss) > 1
            ):
                ratio = 0.8
                spike_loss = loss_reduced["lm loss"] * ratio
                loss_reduced.update(spike_loss=spike_loss)
                logger.info(
                    f' current loss {loss_reduced["lm loss"]}, last loss: {self.last_loss}, spike_loss: {spike_loss}'
                )
            self.last_loss = loss_reduced["lm loss"]
            res["loss_to_log"] = loss_reduced
            res["spike_loss_ratio"] = ratio
            # Just for testing spike loss
            ##################################
        return res


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
        # print_rank_last(f"---> callback: {logs}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AtorchTrainingArgs))
    if len(sys.argv) == 2:
        # If we pass only one argument to the script and it's the path to a json or yaml file,
        # let's parse it to get our arguments.
        if sys.argv[1].endswith(".json"):
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):

            def resolve_env_vars(loader, node):
                value = node.value
                if not isinstance(value, str):
                    return value

                # parse ${VAR:-default_value}
                pattern = re.compile(r"\$\{([^:}]+)(?::-(.+?))?\}")

                def replacer(match):
                    var_name = match.group(1)  # 环境变量名
                    default_value = match.group(2)  # 默认值
                    # 返回环境变量值或默认值
                    print(var_name, os.getenv(var_name, default_value))
                    return os.getenv(var_name, default_value)

                return pattern.sub(replacer, value)

            yaml.SafeLoader.add_constructor("!ENV", resolve_env_vars)

            model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    train_valid_test_datasets_provider.is_distributed = True

    training_args.extra_configs["custom_model_provider_function"] = model_provider
    training_args.extra_configs["custom_megatron_dataloaders_provider_function"] = partial(
        build_train_valid_test_data_iterators, train_valid_test_datasets_provider
    )
    training_args.extra_configs["custom_train_step_class"] = GPTTrainStep

    # Initialize our Trainer
    trainer = AtorchTrainerV2(
        args=training_args,
        callbacks=[CustomCallback()],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()

        metrics = train_result.metrics  # noqa F401

    # max_train_samples = (
    #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    # )
    # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()  # noqa: F841

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # try:
        #     perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        # metrics["perplexity"] = perplexity

        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
