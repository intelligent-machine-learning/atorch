from typing import List, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from dualpipe import DualPipe, set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.utils import WeightGradStore, run_backward

class MoEConfig:
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
        z_loss_coeff: float = 0.0001
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_aux_loss_coeff = aux_loss_coeff
        self.moe_z_loss_coeff = z_loss_coeff
        self.initializer_range = 0.02

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

class MoERouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.moe_aux_loss_coeff = config.moe_aux_loss_coeff
        self.moe_z_loss_coeff = config.moe_z_loss_coeff

    def apply_z_loss(self, logits):
        if self.moe_z_loss_coeff > 0 and self.training:
            z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * self.moe_z_loss_coeff
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_load_balancing_loss(self, router_probs, tokens_per_expert):
        if self.moe_aux_loss_coeff > 0 and self.training:
            # 计算每个专家的负载
            num_tokens = router_probs.shape[0]
            expert_load = router_probs.sum(0) / num_tokens
            # 理想负载
            target_load = torch.ones_like(expert_load) / self.num_experts
            aux_loss = F.mse_loss(expert_load, target_load) * self.moe_aux_loss_coeff
            router_probs = MoEAuxLossAutoScaler.apply(router_probs, aux_loss)
        return router_probs

    def forward(self, x: torch.Tensor):
        router_logits = self.classifier(x)
        router_logits = self.apply_z_loss(router_logits)
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        tokens_per_expert = torch.zeros(self.num_experts, device=router_probs.device)
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(0, top_k_indices[..., k].view(-1), 
                                        torch.ones_like(top_k_indices[..., k].view(-1), dtype=torch.float))
        

        top_k_probs = self.apply_load_balancing_loss(router_probs, tokens_per_expert)
        
        return top_k_probs, top_k_indices

class ExpertLinear(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x)) 
        up = self.up_proj(x)
        activated = gate * up
        return self.down_proj(activated)

class MoEPipelineStage(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = MoERouter(config)
        self.experts = nn.ModuleList([
            ExpertLinear(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])
        
    def forward(self, x: torch.Tensor, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x = x[0]
            
        batch_size, seq_len, hidden_size = x.shape
        
        router_probs, expert_indices = self.router(x)
        
        flat_x = x.view(-1, hidden_size)
        
        combined_output = torch.zeros_like(flat_x)
        for k in range(self.config.top_k):
            expert_index = expert_indices[..., k]
            prob = router_probs[..., k]
            
            flat_expert_index = expert_index.view(-1)
            flat_prob = prob.view(-1)
            
            for i in range(self.config.num_experts):
                mask = (flat_expert_index == i)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[i](expert_input)
                    combined_output[mask] += expert_output * flat_prob[mask].unsqueeze(-1)
        
        output = combined_output.view(batch_size, seq_len, hidden_size)
        return output, router_probs.new_zeros(1)  

    @classmethod
    def overlapped_forward_backward(
        cls,
        module0: "MoEPipelineStage",
        inputs0: List[torch.Tensor],
        criterion0: Optional[Callable],
        labels0: Optional[List[torch.Tensor]],
        module1: "MoEPipelineStage",
        loss1: Optional[torch.Tensor],
        outputs1: Optional[List[torch.Tensor]],
        output_grads1: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        outputs0, router_loss0 = module0(inputs0[0] if isinstance(inputs0, (list, tuple)) else inputs0)
        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        

        if criterion0 is not None:
            task_loss0 = criterion0(*outputs0, *labels0)
            loss0 = task_loss0 + router_loss0
        else:
            loss0 = None


        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        else:
            run_backward(outputs1, output_grads1)

        return outputs0, loss0

def moe_criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return F.mse_loss(output, target)

def ref_step(x, l, model, chunks):
    ys, losses = [], []
    for micro_x, micro_l in zip(x.chunk(chunks), l.chunk(chunks)):

        micro_y, router_loss = model(micro_x)  
        task_loss = moe_criterion(micro_y, micro_l)  
        loss = task_loss + router_loss  
        loss.backward()  
        ys.append(micro_y)  
        losses.append(loss)  
    y = torch.cat(ys, 0)
    loss = torch.stack(losses)
    return loss, y


def cal_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff

def main(rank, pp_size):

    is_first_rank = rank == 0
    is_last_rank = rank == pp_size - 1
    dist.init_process_group(backend='nccl', init_method="env://", world_size=pp_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    torch.manual_seed(233)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


    num_chunks = 20
    micro_batch_size = 3
    seq_len = 256
    hidden_size = 512
    num_experts = 4  
    top_k = 2       
    

    moe_config = MoEConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_coeff=0.01,
        z_loss_coeff=0.0001
    )
    

    if is_first_rank:
        print(f"{pp_size=}, {num_chunks=}, {seq_len=}, {hidden_size=}, {num_experts=}, {top_k=}", flush=True)
    set_p2p_tensor_shapes([(micro_batch_size, seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.float32)


    full_modules = nn.Sequential(*[MoEPipelineStage(moe_config) for _ in range(pp_size)])
    full_x = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)
    full_l = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)


    loss_ref, output_ref = ref_step(full_x, full_l, full_modules, num_chunks)
    print(loss_ref)
    print(output_ref)


def test_dualpipe(ngpus):

    torch.multiprocessing.spawn(main, args=(ngpus,), nprocs=ngpus, daemon=True)

def test_moe_basic():

    hidden_size = 512
    batch_size = 4
    seq_len = 32
    

    config = MoEConfig(
        hidden_size=hidden_size,
        num_experts=4,
        top_k=2,
        aux_loss_coeff=0.01,
        z_loss_coeff=0.0001
    )
    

    model = MoEPipelineStage(config)
    model.cuda()

    x = torch.randn(batch_size, seq_len, hidden_size).cuda()
    target = torch.randn(batch_size, seq_len, hidden_size).cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train for a few steps
    print("Starting test training...")
    for step in range(5):
        optimizer.zero_grad()
        output, router_loss = model(x)
        task_loss = F.mse_loss(output, target)
        loss = task_loss + router_loss
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item():.4f}, Task Loss: {task_loss.item():.4f}, Router Loss: {router_loss.item():.4f}")
    
    print("Basic functionality test completed!")

if __name__ == "__main__":
    # Run basic test first
    print("Running basic MoE model test...")
    test_moe_basic()
    
