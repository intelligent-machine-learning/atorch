from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import fbgemm_gpu  # noqa

    fbgemm_available = True
except Exception:
    fbgemm_available = False


class ScaleType(Enum):
    TENSORWISE = "TENSORWISE"
    AXISWISE = "AXISWISE"
    TILEWISE = "TILEWISE"  # requires scale_args for tile length (1d)
    BLOCKWISE = "BLOCKWISE"  # requires scale_args for block size (1d or 2d)


class ScaleComputMethod(Enum):
    # method for quantization and for fp8 computation
    DEFAULT = "DEFAULT"
    PYTORCH = "PYTORCH"  # use pytorch native ops
    TRITON = "TRITON"  # use triton ops
    CUDA = "CUDA"  # use cuda kernels
    FBGEMM = "FBGEMM"  # use fbgemm_gpu ops


@dataclass
class QuantizeParams:
    """
    use_fp8:bool for whether use fp8
    use_fast_accum: if use_fast_accum in scaled_mm
    fp8_meta: a dict with str as key, and key as a 3-item tuple for (fp8_dtype, scale type, scale args)
    For Linear, the keys are input, weight, and grad
    compute_method: 2-item list for methods used for quantization and for fp8 computation, if None, use default.
    """

    use_fp8: bool = True
    use_fast_accum: bool = True
    fp8_meta: Dict[str, Tuple[torch.dtype, ScaleType, Any]] = field(default_factory=lambda: {})
    compute_method: Optional[List[ScaleComputMethod]] = None

    def get_fp8_type(self, name):
        return self.fp8_meta[name][0]

    def get_scale_type(self, name):
        return self.fp8_meta[name][1]

    def get_scale_args(self, name):
        return self.fp8_meta[name][2]

    def get_quantization_compute_method(self):
        if self.compute_method is None:
            return ScaleComputMethod.DEFAULT
        return self.compute_method[0]

    def get_fp8_compute_method(self):
        if self.compute_method is None:
            return ScaleComputMethod.DEFAULT
        return self.compute_method[1]


def get_quantize_params(scale_method, quantization_method, compute_method, block_size=128):
    scale_methods = None
    dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)
    scale_args = None
    if isinstance(scale_method, str):
        if scale_method == "tileblock":
            scale_methods = (ScaleType.TILEWISE, ScaleType.BLOCKWISE, ScaleType.TILEWISE)
            dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn)
        else:
            scale_method = ScaleType(scale_method.upper())
    if scale_methods is None:
        if isinstance(scale_method, tuple):
            scale_methods = []
            for m in scale_method:
                method = ScaleType(m.upper()) if isinstance(m, str) else m
                assert isinstance(scale_method, ScaleType)
                scale_methods.append(method)
            scale_methods = tuple(scale_methods)
        else:
            assert isinstance(scale_method, ScaleType)
            scale_methods = (scale_method, scale_method, scale_method)
    if ScaleType.TILEWISE in scale_methods or ScaleType.BLOCKWISE in scale_methods:
        scale_args = block_size

    if isinstance(quantization_method, str):
        quantization_method = ScaleComputMethod(quantization_method.upper())
    if isinstance(compute_method, str):
        compute_method = ScaleComputMethod(compute_method.upper())

    if compute_method == ScaleComputMethod.DEFAULT and scale_args is not None:
        # for tileblock, now only supported by triton fp8 gemm.
        compute_method = ScaleComputMethod.TRITON

    qparams = QuantizeParams(compute_method=[quantization_method, compute_method])
    qparams.fp8_meta["input"] = (dtypes[0], scale_methods[0], scale_args)
    qparams.fp8_meta["weight"] = (dtypes[1], scale_methods[1], scale_args)
    qparams.fp8_meta["grad"] = (dtypes[2], scale_methods[2], scale_args)

    return qparams


def get_linear_tensorwise_quantize_params():
    # TENSORWISE for all, and e5m2 for bwd_g, e4m3 for others.
    qparams = QuantizeParams()
    qparams.fp8_meta["input"] = (torch.float8_e4m3fn, ScaleType.TENSORWISE, None)
    qparams.fp8_meta["weight"] = (torch.float8_e4m3fn, ScaleType.TENSORWISE, None)
    qparams.fp8_meta["grad"] = (torch.float8_e5m2, ScaleType.TENSORWISE, None)
    return qparams


def get_linear_axiswise_quantize_params():
    # AXISWISE for all, and e5m2 for bwd_g, e4m3 for others.
    qparams = QuantizeParams()
    qparams.fp8_meta["input"] = (torch.float8_e4m3fn, ScaleType.AXISWISE, None)
    qparams.fp8_meta["weight"] = (torch.float8_e4m3fn, ScaleType.AXISWISE, None)
    qparams.fp8_meta["grad"] = (torch.float8_e5m2, ScaleType.AXISWISE, None)
    return qparams


def get_linear_tileblock_quantize_params(block_size):
    # tile for input, block for weight, and e5m2 for all.
    qparams = QuantizeParams()
    qparams.fp8_meta["input"] = (torch.float8_e4m3fn, ScaleType.TILEWISE, block_size)
    qparams.fp8_meta["weight"] = (torch.float8_e4m3fn, ScaleType.BLOCKWISE, block_size)
    qparams.fp8_meta["grad"] = (torch.float8_e4m3fn, ScaleType.TILEWISE, block_size)
    qparams.compute_method = (ScaleComputMethod.TRITON, ScaleComputMethod.TRITON)
    return qparams


class Fp8Quantization:
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max if hasattr(torch, "float8_e4m3fn") else 448.0
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max if hasattr(torch, "float8_e5m2") else 57344.0
    EPS: float = 1e-12

    @staticmethod
    def _amax_to_scale(amax: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            amax = amax.float()
            if float8_dtype == torch.float8_e4m3fn:
                res = Fp8Quantization.E4M3_MAX_POS / torch.clamp(amax, min=Fp8Quantization.EPS)
            else:  # e5m2
                res = Fp8Quantization.E5M2_MAX_POS / torch.clamp(amax, min=Fp8Quantization.EPS)
            return res

    @staticmethod
    def quantize_tensorwise(
        x: torch.Tensor, float8_dtype: torch.dtype, method: ScaleComputMethod = ScaleComputMethod.DEFAULT
    ):
        if method == ScaleComputMethod.DEFAULT or method == ScaleComputMethod.PYTORCH:
            return Fp8Quantization.quantize_tensorwise_pt(x, float8_dtype)
        else:
            assert 0, f"{ScaleComputMethod} not implemented"

    @staticmethod
    def quantize_axiswise(
        x: torch.Tensor, float8_dtype: torch.dtype, dim=1, method: ScaleComputMethod = ScaleComputMethod.DEFAULT
    ):
        if (
            (method == ScaleComputMethod.DEFAULT and not fbgemm_available)
            or method == ScaleComputMethod.PYTORCH
            or x.dtype == torch.float16
        ):
            return Fp8Quantization.quantize_axiswise_pt(x, float8_dtype, dim)
        elif (method == ScaleComputMethod.DEFAULT and fbgemm_available) or ScaleComputMethod.FBGEMM:
            # fbgemm only supports bf16, fp32, not fp16
            return Fp8Quantization.quantize_axiswise_fbgemm(x, float8_dtype, dim)
        else:
            assert 0, f"{ScaleComputMethod} not implemented"

    @staticmethod
    def quantize_tilewise(
        x: torch.Tensor,
        float8_dtype: torch.dtype,
        block_size,
        dim=1,
        method: ScaleComputMethod = ScaleComputMethod.DEFAULT,
    ):
        if method == ScaleComputMethod.DEFAULT or method == ScaleComputMethod.TRITON:
            return Fp8Quantization.quantize_tilewise_triton(x, float8_dtype, block_size=block_size, dim=dim)
        else:
            assert 0, f"{ScaleComputMethod} not implemented"

    @staticmethod
    def quantize_blockwise(
        x: torch.Tensor, float8_dtype: torch.dtype, block_size, method: ScaleComputMethod = ScaleComputMethod.DEFAULT
    ):
        if method == ScaleComputMethod.DEFAULT or method == ScaleComputMethod.TRITON:
            # Only support square block shape
            assert isinstance(block_size, int) or block_size[0] == block_size[1]
            bsize = block_size if isinstance(block_size, int) else block_size[0]
            return Fp8Quantization.quantize_blockwise_triton(x, float8_dtype, block_size=bsize)
        else:
            assert 0, f"{ScaleComputMethod} not implemented"

    @staticmethod
    def quantize_tensorwise_pt(x: torch.Tensor, float8_dtype: torch.dtype):
        amax = torch.max(torch.abs(x))
        scale = Fp8Quantization._amax_to_scale(amax, float8_dtype)
        x_fp8 = (x * scale).to(float8_dtype)
        inverse_scale = scale.reciprocal()
        return x_fp8, inverse_scale

    @staticmethod
    def quantize_axiswise_pt(x: torch.Tensor, float8_dtype: torch.dtype, dim=1):
        # set dim=1 for rowwise, dim=0 for colwise.
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values
        scale = Fp8Quantization._amax_to_scale(amax, float8_dtype)
        x_fp8 = (x * scale).to(float8_dtype)
        inverse_scale = scale.reciprocal()
        return x_fp8, inverse_scale

    @staticmethod
    def quantize_axiswise_fbgemm(x: torch.Tensor, float8_dtype: torch.dtype, dim=1):
        if not fbgemm_available:
            assert 0, "fbgemm_gpu not installed"
        if dim == 1:
            return torch.ops.fbgemm.quantize_fp8_per_row(x.contiguous(), output_dtype=float8_dtype)
        else:  # dim == 0
            # quantize_fp8_per_col perf is bad, use quantize_fp8_per_row with transpose
            # assert float8_dtype == torch.float8_e4m3fn
            # return torch.ops.fbgemm.quantize_fp8_per_col(x)
            res = torch.ops.fbgemm.quantize_fp8_per_row(x.t().contiguous(), output_dtype=float8_dtype)
            return (res[0].t(), res[1])

    @staticmethod
    def quantize_tilewise_triton(x: torch.Tensor, float8_dtype: torch.dtype, block_size=128, dim=1):
        # Now only support dim=1
        assert dim == 1
        from .triton_kernel import tile_quant

        return tile_quant(x, dtype=float8_dtype, block_size=block_size)

    @staticmethod
    def quantize_blockwise_triton(x: torch.Tensor, float8_dtype: torch.dtype, block_size=128):
        from .triton_kernel import block_quant

        return block_quant(x, dtype=float8_dtype, block_size=block_size)


def get_fp8_quantize_underflows(
    data,
    fp8_dtype=None,
    tensorwise_required=True,
    rowwise_required=True,
    colwise_required=True,
    tilewise_required=True,
    blockwise_required=True,
    block_size=128,
    quantize_method="DEFAULT",
):
    if fp8_dtype is None:
        fp8_dtype = torch.float8_e4m3fn
    data = data.view(-1, data.shape[-1])
    total_zero = (data == 0).sum().item()
    total_nonzero = data.numel() - total_zero
    results = {}
    if tensorwise_required:
        tensor_fp8, _ = Fp8Quantization.quantize_tensorwise(data, fp8_dtype)
        tensor_zero = (tensor_fp8 == 0).sum().item()
        tenor_underflow = (tensor_zero - total_zero) / total_nonzero * 100.0
        results["tensorwise"] = tenor_underflow
    if rowwise_required:
        row_fp8, _ = Fp8Quantization.quantize_axiswise(
            data, fp8_dtype, dim=1, method=ScaleComputMethod(quantize_method)
        )
        row_zero = (row_fp8 == 0).sum().item()
        row_underflow = (row_zero - total_zero) / total_nonzero * 100.0
        results["rowwise"] = row_underflow
    if colwise_required:
        col_fp8, _ = Fp8Quantization.quantize_axiswise(
            data, fp8_dtype, dim=0, method=ScaleComputMethod(quantize_method)
        )
        col_zero = (col_fp8 == 0).sum().item()
        col_underflow = (col_zero - total_zero) / total_nonzero * 100.0
        results["colwise"] = col_underflow
    if tilewise_required and data.size(1) % block_size == 0:
        tile_fp8, _ = Fp8Quantization.quantize_tilewise(
            data, fp8_dtype, block_size, method=ScaleComputMethod(quantize_method)
        )
        tile_zero = (tile_fp8 == 0).sum().item()
        tile_underflow = (tile_zero - total_zero) / total_nonzero * 100.0
        results["tilewise"] = tile_underflow
    if blockwise_required and data.size(0) % block_size == 0 and data.size(1) % block_size == 0:
        block_fp8, _ = Fp8Quantization.quantize_blockwise(
            data, fp8_dtype, block_size, method=ScaleComputMethod(quantize_method)
        )
        block_zero = (block_fp8 == 0).sum().item()
        block_underflow = (block_zero - total_zero) / total_nonzero * 100.0
        results["blockwise"] = block_underflow
    return results
