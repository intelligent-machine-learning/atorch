import functools
import inspect

import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from atorch.common.util_func import data_to_device
from atorch.distributed.distributed import get_device_mesh
from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE
from atorch.modules.transformer.rmsnorm import AtorchRmsNorm
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload

MOE_IMPLEMENTATION_TYPE = None
MOE_TOKEN_DISPATCHER_TYPE = None


def set_global_variable_from_args(args):
    global MOE_IMPLEMENTATION_TYPE
    global MOE_TOKEN_DISPATCHER_TYPE
    MOE_IMPLEMENTATION_TYPE = args.moe_implementation_type
    MOE_TOKEN_DISPATCHER_TYPE = args.moe_token_dispatcher_type


class MoEAuxLossAutoScaler(torch.autograd.Function):
    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


class TopNRouter(torch.nn.Module):
    """
    This implementation is equivalent to the standard
    TopN MoE with full capacity without dropp tokens.
    """

    def __init__(self, config, top_k=2, norm_prob=True):
        super().__init__()
        self.num_experts = config.num_experts
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.top_k = top_k
        self.norm_prob = norm_prob
        self.moe_aux_loss_coeff = config.moe_aux_loss_coeff if hasattr(config, "moe_aux_loss_coeff") else None
        self.moe_z_loss_coeff = config.moe_z_loss_coeff if hasattr(config, "moe_z_loss_coeff") else None

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.classifier(hidden_states)
        # router_logits = torch.rand_like(router_logits)

        return router_logits

    def _route_tokens(self, router_logits: torch.Tensor):
        original_shape = router_logits.shape[:-1] + (self.top_k,)
        router_logits = router_logits.view(-1, self.num_experts)
        scores, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_map = torch.zeros_like(router_logits).int().scatter(1, top_indices, 1).bool()
        tokens_per_expert = topk_map.sum(dim=0)

        scores = self.apply_load_balancing_loss(router_logits, tokens_per_expert, activation=scores)
        return scores.view(original_shape), top_indices.view(original_shape)

    def apply_z_loss(self, logits):
        if self.moe_z_loss_coeff is not None and self.training:
            z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * self.moe_z_loss_coeff
            # TODO: save z_loss to loss tracker
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_load_balancing_loss(self, logits, indices, activation):
        if self.moe_aux_loss_coeff is not None and self.training:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            num_tokens = scores.shape[0]
            num_experts = scores.shape[1]
            aggregated_probs_per_expert = scores.sum(dim=0)
            aux_loss = torch.sum(aggregated_probs_per_expert * indices) * (
                num_experts * self.moe_aux_loss_coeff / (num_tokens * num_tokens * self.top_k)
            )
            # TODO: save balance loss to loss tracker
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self._compute_router_probabilities(hidden_states)
        logits = self.apply_z_loss(router_logits)
        router_probs, topk_experts_index = self._route_tokens(logits)
        return router_probs, topk_experts_index


class _MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.silu

    def forward_w1_fn(self, x):
        return self.act_fn(self.gate_proj(x))

    def forward_w2_fn(self, x, w1_fn_input_x):
        x = x * self.up_proj(w1_fn_input_x)
        x = self.down_proj(x)
        return x

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class _SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        global MOE_IMPLEMENTATION_TYPE
        global MOE_TOKEN_DISPATCHER_TYPE

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_shared_expert = config.num_shared_expert
        self.intermediate_size = config.intermediate_size
        self.use_expert_parallelism = config.ep_size > 1

        self.shared_expert_overlapping = config.shared_expert_overlapping

        self.router = TopNRouter(config, config.top_k)

        self.experts = Grouped_GEMM_MoE(
            hidden_size=config.hidden_size,
            expert_intermediate_size=self.intermediate_size,
            output_dropout_prob=0.1,
            num_experts=self.num_experts,
            topk=config.top_k,
            use_swiglu=True,
            use_bias=False,
            initializer_range=config.initializer_range,
            use_expert_parallelism=self.use_expert_parallelism,
            expert_parallel_group=None,
            implementation_type=MOE_IMPLEMENTATION_TYPE,
            token_dispatcher_type=MOE_TOKEN_DISPATCHER_TYPE,
        )

        if config.num_shared_expert > 0:
            self.shared_experts = _MLP(config, intermediate_size=self.intermediate_size * config.num_shared_expert)
        else:
            self.shared_experts = None

    def forward(self, hidden_states):
        router_probs, top_expert_index = self.router(hidden_states)
        identify = hidden_states

        if self.shared_experts is not None and self.use_expert_parallelism:
            hidden_shape = hidden_states.shape
            temp_hidden_states = hidden_states.view(-1, hidden_shape[-1])
            if self.shared_expert_overlapping:
                shared_experts_fn = (self.shared_experts.forward_w1_fn, self.shared_experts.forward_w2_fn)
                se_fn2_additional_input = temp_hidden_states
            else:
                shared_experts_fn = (None, None)
                se_fn2_additional_input = None

            hidden_states = self.experts(
                hidden_states,
                router_probs,
                top_expert_index,
                *shared_experts_fn,
                se_fn2_additional_input=se_fn2_additional_input,
            )

            if not self.shared_expert_overlapping:
                se_output = self.shared_experts(temp_hidden_states)
                se_output = se_output.view(hidden_shape)
                hidden_states = hidden_states + se_output
        else:
            hidden_states = self.experts(hidden_states, router_probs, top_expert_index)
        hidden_states = hidden_states.to(identify.dtype)

        if self.shared_experts is not None and not self.use_expert_parallelism:
            hidden_states = hidden_states + self.shared_experts(identify)

        return hidden_states


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
        if Version(transformers.__version__) >= Version("4.48.0"):
            # 4.48.0 之后，传入 decoder_layer 的 position_embeddings 不能为 None。
            self.rotary_emb = LlamaRotaryEmbedding(config=config)
        else:
            self.rotary_emb = None
        self.layers = torch.nn.ModuleDict()

        for layer_idx_cur_stage in range(self.layer_num):
            if start_layer_idx is not None:
                global_layer_idx = start_layer_idx + layer_idx_cur_stage
            else:
                global_layer_idx = layer_idx_cur_stage

            if requires_layer_idx:
                self.layers[str(global_layer_idx)] = LlamaDecoderLayer(config, layer_idx=None)
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
        position_embeddings = self.rotary_emb and self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        if self.norm is not None and self.lm_head is not None:
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

        return hidden_states


def patch_llama(args):
    modeling_llama.LlamaMLP = _SparseMLP
    if not args.not_use_atorch_rmsnorm:
        modeling_llama.LlamaRMSNorm = AtorchRmsNorm


def llama_loss_func(inputs, output):
    return output.loss


def pp_llama_loss_func(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def prepare_input(data, device):
    return data_to_device(data, device)


def get_model(config, meta_init=False):
    if meta_init:
        with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
            model = LlamaForCausalLM(config)
    else:
        model = LlamaForCausalLM(config)
    return model


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
def collate_sentences_lm(samples, input_name, label_name):
    if len(samples) == 0:
        return {}

    if len(samples[0]) == 0:
        return {}

    src_tokens = torch.stack([s["source"] for s in samples], 0)
    tgt_tokens = torch.stack([s["target"] for s in samples], 0)

    batch = {
        input_name: src_tokens,
        label_name: tgt_tokens,
    }
    return batch


# copy from pytorch/benchmarks/distributed/pipeline/benchmark_dataset.py with some modification
class BenchmarkLMDataset(Dataset):
    def __init__(
        self,
        vocab_size=10000,
        max_source_positions=1024,
        total_samples=10000,
        null_dataset=False,
    ):
        self.vocab_size = vocab_size
        self.max_source_positions = max_source_positions
        self.total_samples = total_samples
        self.null_dataset = null_dataset

    def __getitem__(self, index):
        if self.null_dataset:
            return {}
        length = self.max_source_positions
        source = torch.randint(1, self.vocab_size, (length,))
        target = source.clone()
        return {
            "source": source,
            "target": target,
        }

    def __len__(self):
        return self.total_samples


def get_dataset(seq_length=128, vocab_size=32000, datasize=1000, null_dataset=False):
    return BenchmarkLMDataset(
        vocab_size=vocab_size, max_source_positions=seq_length, total_samples=datasize, null_dataset=null_dataset
    )


def get_dataloader(dataset, batch_size, rank, dp_size, use_distributed):
    dataloader_args = {"batch_size": batch_size, "drop_last": True, "shuffle": False, "num_workers": 2}
    input_name = "input_ids"
    label_name = "labels"
    dataloader_args["collate_fn"] = functools.partial(
        collate_sentences_lm, input_name=input_name, label_name=label_name
    )
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=dp_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, **dataloader_args)
    return dataloader


def optim_grouped_param_func(model):
    no_decay = "bias"
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if no_decay not in n],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if no_decay in n],
            "weight_decay": 0.0,
        },
    ]
    return parameters


def moe_fsdp2_policy_fn(module, nested_fsdp2=False):
    data_device_mesh = get_device_mesh("data")
    if nested_fsdp2:
        expert_fsdp_device_mesh = get_device_mesh("expert_fsdp")
        cls_to_fsdp_device_mesh = {LlamaDecoderLayer: data_device_mesh, Grouped_GEMM_MoE: expert_fsdp_device_mesh}
    else:
        cls_to_fsdp_device_mesh = {LlamaDecoderLayer: data_device_mesh}

    if module.__class__ in cls_to_fsdp_device_mesh:
        dm = cls_to_fsdp_device_mesh[module.__class__]
        return {"device_mesh": dm}
    return False
