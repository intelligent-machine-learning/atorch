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

import copy
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

import torch
import yaml  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoTokenizer, HfArgumentParser, TrainerCallback
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.atorch_trainer_v2 import AtorchTrainerV2
from atorch.trainer.megatron import MegatronTrainStep
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    import megatron.legacy.model
    from megatron.core import mpu
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training import get_args
    from megatron.training import get_tokenizer as megatron_get_tokenizer
    from megatron.training import print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
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
    dataset_path: Optional[str] = field(default=None, metadata={"help": "A dir containing dataset with .arrow format."})
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

    drop_last: bool = field(default=False, metadata={"help": "Whether to drop last batch in dataloader."})

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


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


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
            # batch = get_batch_on_this_tp_rank(data_iterator)

            # # slice batch along sequence dimension for context parallelism
            # batch = get_batch_on_this_cp_rank(batch)

            batch = sft_get_batch_on_this_tp_rank(data_iterator)

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


class MegatronDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):

        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.rank = mpu.get_data_parallel_rank()
        super().__init__(
            dataset, num_replicas=self.dp_world_size, rank=self.rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# copy from llama-recipes
# https://github.com/facebookresearch/llama-recipes/blob/405255c/ft_datasets/alpaca_dataset.py
class InstructionDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, partition="train", max_words=30):
        self.ann = json.load(open(dataset_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        args = get_args()

        logger.info(f"global step {args.global_step} rank {args.rank}, index {index}")

        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]

        ############ Original InstructionDataset
        # prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        # example = self.tokenizer.encode(example)
        # example.append(self.tokenizer.eos_token_id)

        ############ Add by jinshi.cl, for Megatron ############
        prompt = torch.tensor(self.tokenizer.tokenize(prompt), dtype=torch.int64)
        example = self.tokenizer.tokenize(example)
        example.append(self.tokenizer.eos_id)

        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        ############ Original InstructionDataset
        # return {
        #     "input_ids": example,
        #     "labels": labels,
        #     "attention_mask": example_mask,
        # }

        ############ Add by jinshi.cl, for Megatron ############
        return {
            "tokens": example,
            "labels": labels,
            "loss_mask": label_mask,
        }


def get_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def build_train_valid_test_data_iterators_for_sft(model_args, data_args, training_args: AtorchTrainingArgs):
    from megatron.training.global_vars import get_args

    args = get_args()

    # tokenizer = get_tokenizer(model_args)
    tokenizer = megatron_get_tokenizer()

    if not is_dataset_built_on_rank():
        print(f"dataset: rank {torch.distributed.get_rank()}, build empty dataloader.")
        return None, None, None
    else:
        train_dataset = InstructionDataset(
            data_args.dataset_path,
            tokenizer,
            partition="train",
            max_words=data_args.block_size,
        )

        eval_dataset = InstructionDataset(
            data_args.dataset_path,
            tokenizer,
            partition="eval",
            max_words=data_args.block_size,
        )

        batch_size = args.micro_batch_size

        train_dataloader = DataLoader(
            train_dataset,
            sampler=MegatronDistributedSampler(
                train_dataset, shuffle=True, seed=training_args.seed, drop_last=data_args.drop_last
            ),
            batch_size=batch_size,
            pin_memory=True,
            drop_last=data_args.drop_last,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=MegatronDistributedSampler(eval_dataset),
            batch_size=batch_size,
            pin_memory=True,
            drop_last=data_args.drop_last,
        )

        print(
            f"[Rank {torch.distributed.get_rank()}] train_dataloader_type {type(train_dataloader)} train_dataset "
            f"{len(train_dataset)} train_dataloader {len(train_dataloader)} eval_dataset {len(eval_dataset)}",
            flush=True,
        )
        return train_dataloader, eval_dataloader, None


def sft_get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group()
            )

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": None,
            "position_ids": None
            # 'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),  # noqa: E501
            # 'position_ids': data["position_ids"].cuda(non_blocking = True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch["tokens"])
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            # _broadcast(batch['attention_mask'])
            # _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch["tokens"])
            # _broadcast(batch['attention_mask'])
            # _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            # _broadcast(batch['attention_mask'])

    else:

        tokens = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.int64, device=torch.cuda.current_device()
        )
        loss_mask = torch.empty(
            (args.micro_batch_size, args.seq_length), dtype=torch.float32, device=torch.cuda.current_device()
        )
        # if args.create_attention_mask_in_dataloader:
        #     attention_mask=torch.empty(
        #         (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()  # noqa: E501
        #     )
        # else:
        #     attention_mask=None
        # position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())  # noqa: E501

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            # _broadcast(attention_mask)
            # _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            # _broadcast(attention_mask)
            # _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            # position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            # _broadcast(attention_mask)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": None,
            "position_ids": None
            # 'attention_mask': attention_mask,
            # 'position_ids': position_ids
        }

    return batch


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

    assert training_args.finetune_type is not None, "finetune_type should be set!"

    training_args.extra_configs["custom_model_provider_function"] = model_provider
    training_args.extra_configs["custom_train_step_class"] = GPTTrainStep
    training_args.extra_configs["custom_megatron_dataloaders_provider_function"] = partial(
        build_train_valid_test_data_iterators_for_sft, model_args, data_args, training_args
    )

    if model_args.model_name_or_path is not None:
        tracker_filename = os.path.join(model_args.model_name_or_path, "latest_checkpointed_iteration.txt")
        assert os.path.exists(
            tracker_filename
        ), f"latest_checkpointed_iteration.txt should be in {model_args.model_name_or_path}."
        with open(tracker_filename, "r") as f:
            metastring = f.read().strip()
            assert (
                metastring == "release"
            ), f"The content of {tracker_filename} should be 'release', but got {metastring}."
        pretrained_model_dir = os.path.join(model_args.model_name_or_path, "release")
        assert os.path.isdir(pretrained_model_dir), f"pretrained model dir {pretrained_model_dir} does not exist!"

    # If resume from checkpoint, training_args.resume_from_checkpoint is equal to training_args.output_dir
    if training_args.resume_from_checkpoint is None:
        training_args.resume_from_checkpoint = model_args.model_name_or_path
        logger.info(f"Train from pretrained model {training_args.resume_from_checkpoint}.")
    else:
        assert training_args.resume_from_checkpoint == training_args.output_dir, (
            "training_args.resume_from_checkpoint should be equal to training_args.output_dir, "
            f"but got {training_args.resume_from_checkpoint} and {training_args.output_dir}."
        )
        tracker_filename = os.path.join(training_args.resume_from_checkpoint, "latest_checkpointed_iteration.txt")
        if os.path.exists(tracker_filename):
            with open(tracker_filename, "r") as f:
                metastring = f.read().strip()
                try:
                    iteration = int(metastring)
                    assert iteration > 0
                except ValueError:
                    logger.error(f"iteration value in {tracker_filename} should be greater than 0!")
                    raise
            resumed_ckpt = os.path.join(training_args.resume_from_checkpoint, f"iter_{iteration:07d}")
            logger.info(f"Train from resumed model {resumed_ckpt}.")
        else:
            training_args.resume_from_checkpoint = model_args.model_name_or_path
            logger.info(f"Train from pretrained model {training_args.resume_from_checkpoint}.")

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
