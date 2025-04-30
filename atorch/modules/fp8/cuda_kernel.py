import importlib
import os
from typing import Optional, Tuple

import torch

try:
    fp8_ops_module_name = os.getenv("FP8_GEMM_MODULE_OPS", "kitchen.ops")
    ops = importlib.import_module(fp8_ops_module_name)
except ModuleNotFoundError:
    ops = None


def fp8_cutlass_cublas_ops_available():
    return ops is not None


def tile_quant(
    x: torch.Tensor,
    dtype=torch.float8_e4m3fn,
    block_size: int = 128,
    pow_2_scale: bool = False,
    return_transpose: bool = False,
    use_cublas=False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # return qx, sx, qx_t, sx_t
    eps = 1e-10
    # only 128 is supported for block_size for now
    assert block_size == 128
    return ops.quantize_vector_blockwise(
        x,
        dtype,
        block_size,
        return_transpose=return_transpose,
        eps=eps,
        pow_2_scale=pow_2_scale,
        backend=ops.Backend.CUBLAS if use_cublas else ops.Backend.CUTLASS,
    )


def block_quant(
    x: torch.Tensor,
    dtype=torch.float8_e4m3fn,
    block_size: int = 128,
    pow_2_scale: bool = False,
    return_transpose: bool = False,
    use_cublas=False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    # return qx, sx, qx_t, sx_t
    eps = 1e-10
    # only 128 is supported for block_size for now
    assert block_size == 128
    return ops.quantize_square_blockwise(
        x,
        dtype,
        block_size,
        return_transpose=return_transpose,
        eps=eps,
        pow_2_scale=pow_2_scale,
        backend=ops.Backend.CUBLAS if use_cublas else ops.Backend.CUTLASS,
    )


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, dtype: torch.types, use_cublas=False
):
    is_a_1d_scaled = a.numel() == a_s.numel() * 128
    is_b_1d_scaled = b.numel() == b_s.numel() * 128
    gemm_op = ops.fp8_gemm_blockwise if use_cublas else ops.cutlass_gemm_fp8_blockwise
    return gemm_op(
        a,
        a_s,
        b,
        b_s,
        dtype,
        out=None,
        accumulate=False,
        use_split_accumulator=True,
        is_a_1d_scaled=is_a_1d_scaled,
        is_b_1d_scaled=is_b_1d_scaled,
    )
