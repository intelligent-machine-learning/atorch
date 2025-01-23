import inspect
import os

import torch
from torch import nn

import atorch
from atorch.distributed.distributed import create_parallel_group
from atorch.utils.version import torch_version

if torch_version() >= (2, 4, 0):  # type: ignore
    from torch.distributed.pipelining import SplitPoint
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaRMSNorm
else:
    pipe_split = None  # type: ignore
    SplitPoint = object
    LlamaModel = object
    LlamaDecoderLayer = None
    LlamaRMSNorm = None


# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


# Multi-MLP model
class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # For testing purpose only, this should be defined by user
        self.split_spec = {f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)}

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LlamaChunk(LlamaModel):
    def __init__(self, config, layer_num=None, pre_process=True, post_process=True, start_layer_idx=None):
        nn.Module.__init__(self)
        self.config = config

        requires_layer_idx = "layer_idx" in inspect.signature(LlamaDecoderLayer.__init__).parameters

        if pre_process:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        else:
            self.embed_tokens = None

        if layer_num is None:
            self.layer_num = config.num_hidden_layers
        else:
            self.layer_num = layer_num

        self.layers = torch.nn.ModuleDict()

        for layer_idx_cur_stage in range(self.layer_num):
            if start_layer_idx is not None:
                global_layer_idx = start_layer_idx + layer_idx_cur_stage
            else:
                global_layer_idx = layer_idx_cur_stage

            if requires_layer_idx:
                self.layers[str(global_layer_idx)] = LlamaDecoderLayer(config, layer_idx=global_layer_idx)
            else:
                self.layers[str(global_layer_idx)] = LlamaDecoderLayer(config)

        if post_process:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

    def forward(self, input_ids):
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_ids

        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            hidden_states = hidden_states.to(target_dtype)

        position_ids = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0)
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(hidden_states, position_ids=position_ids)[0]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states


def get_llama_model_chunk(
    model_config,
    seed=123,
    layer_num=None,
    pre_process=True,
    post_process=True,
    start_layer_idx=None,
):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return LlamaChunk(
        config=model_config,
        layer_num=layer_num,
        pre_process=pre_process,
        post_process=post_process,
        start_layer_idx=start_layer_idx,
    )


def llama_loss_func(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def create_pipe_group(rank):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    set_cuda_device = backend == "nccl"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=set_cuda_device)
    assert res
    world_size = torch.distributed.get_world_size()
    gpu_partition = ([("pipe", world_size)], None)
    create_parallel_group(gpu_partition, use_atorch_pipe=True)
