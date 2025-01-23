# Additional Requirements for ATorch Features

Some ATorch features require more Python packages than those listed in ATorch's requirement. Users need to install corresponding requirements when using these features.

## atorch.modules.moe.grouped_gemm_moe.Grouped_GEMM_MoE
- grouped_gemm
- megablocks (if implementation_type="MegaBlocks")

## fp8
- transformer_engine

## atorch.rl
- deepspeed

## ATorch megatron trainer
- megatron

## auto_accelerate fully automatic mode
- pymoo==0.5.0
- GPy

## apex fused kernels
- apex
