import io
from enum import Enum
from typing import Union

import torch

from atorch.common.env import EnvSetting

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.float8_tensor import Float8Tensor
    from transformer_engine.pytorch.fp8 import _default_sf_compute
    from transformer_engine.pytorch.utils import check_dim_for_fp8_exec

    try:
        import transformer_engine_torch as tex
    except (ImportError, ModuleNotFoundError):
        import transformer_engine_extensions as tex  # te version < 1.12

    from transformer_engine.common.recipe import Format
except (ImportError, ModuleNotFoundError):
    print("Transformer_engine required for PrecisionSwitchableLinear.")

_ORIGINAL_TE_LINEAR_BACKWARD_FUNC = None


class LinearPrecision(Enum):
    ORIGINAL = "original"
    FP8 = "fp8"


def _patch_te_linear_backward(enable=True):
    # patch _Linear backward to use non-fp8 for input grad even if input is fp8
    from transformer_engine.pytorch.module.linear import _Linear

    global _ORIGINAL_TE_LINEAR_BACKWARD_FUNC
    if _ORIGINAL_TE_LINEAR_BACKWARD_FUNC is None:
        _ORIGINAL_TE_LINEAR_BACKWARD_FUNC = _Linear.backward

    def patched_linear_backward(ctx, grad_output: torch.Tensor):
        ctx.is_input_fp8 = False
        return _ORIGINAL_TE_LINEAR_BACKWARD_FUNC(ctx, grad_output)

    if enable:
        _Linear.backward = patched_linear_backward
    else:
        _Linear.backward = _ORIGINAL_TE_LINEAR_BACKWARD_FUNC


class _Fp8_Cast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, fp8_dtype, margin):
        if fp8_dtype == tex.DType.kFloat8E4M3:
            fp8_max = Format.E4M3.value.max_fwd
        else:
            fp8_max = Format.E5M2.value.max_fwd

        amax = tensor.abs().max().float()
        one = torch.ones(1, device="cuda")

        scale = _default_sf_compute(amax, one, fp8_max, margin)
        scale_inv = 1.0 / scale

        fp8_tensor = tex.cast_to_fp8(tensor, scale, amax, scale_inv, fp8_dtype)
        fp8_tensor = Float8Tensor(
            data=fp8_tensor,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=scale_inv,
            dtype=tensor.dtype,
        )
        fp8_tensor.requires_grad = tensor.requires_grad
        return fp8_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.dequantize() if isinstance(grad_output, Float8Tensor) else grad_output
        return (grad, None, None)


def fp8_cast(tensor, fp8_dtype, margin=0):
    return _Fp8_Cast.apply(tensor, fp8_dtype, margin)


class PrecisionSwitchableLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        original_linear=None,
        init_precision="fp8",
        pre_cast_input_fp8_current_scaling=False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = device
        self.dtype = dtype
        self.pre_cast_input_fp8_current_scaling = False
        self.fp8_backward_patched = False
        self.set_pre_cast_input_fp8_current_scaling(pre_cast_input_fp8_current_scaling)

        self._precision_modules = {}

        self._precision_modules[LinearPrecision.ORIGINAL] = (
            torch.nn.Linear(in_features, out_features, bias, device, dtype)
            if original_linear is None
            else original_linear
        )

        self._parameters["weight"] = self._precision_modules[LinearPrecision.ORIGINAL].weight
        self._precision_modules[LinearPrecision.ORIGINAL].weight = None
        if self.use_bias:
            self._parameters["bias"] = self._precision_modules[LinearPrecision.ORIGINAL].bias
            self._precision_modules[LinearPrecision.ORIGINAL].bias = None

        self.selection = LinearPrecision(init_precision) if isinstance(init_precision, str) else LinearPrecision

        self._precision_modules[LinearPrecision.FP8] = self._create_fp8_module()

    def _create_fp8_module(self):
        # Backward computation would use transposed weight, so also check transposed weight shape.
        transposed_shape = torch.Size([self.out_features, self.in_features])
        backward_weight = torch.empty(transposed_shape, device="meta")
        assert check_dim_for_fp8_exec(self._parameters["weight"]) and check_dim_for_fp8_exec(
            backward_weight
        ), f"shape [{self.in_features}, {self.out_features}] not compatible with fp8 requirement"

        # te check if device is meta by str.
        if isinstance(self.device, torch.device) and self.device.type == "meta":
            device = "meta"
        else:
            device = self.device

        fp8_module = te.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.use_bias,
            params_dtype=self.dtype,
            device=device,
        )
        # change weight, bias param to None to avoid memory leak.
        fp8_module.weight = None
        if self.use_bias:
            fp8_module.bias = None
        return fp8_module

    def forward(self, input):
        self._pre_forward_assign()
        if (
            self.pre_cast_input_fp8_current_scaling
            and self.selection == LinearPrecision.FP8
            and not isinstance(input, Float8Tensor)
        ):
            input = fp8_cast(input, tex.DType.kFloat8E4M3)
        return self._precision_modules[self.selection](input)

    def _pre_forward_assign(self):
        # parameters may be modified (by fsdp), re-assign weight and bias.
        self._precision_modules[self.selection]._parameters["weight"] = self.weight
        if self.use_bias:
            self._precision_modules[self.selection]._parameters["bias"] = self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"

    def set_pre_cast_input_fp8_current_scaling(self, pre_cast_input_fp8_current_scaling: bool):
        self.pre_cast_input_fp8_current_scaling = pre_cast_input_fp8_current_scaling
        if (
            pre_cast_input_fp8_current_scaling
            and not self.fp8_backward_patched
            and not EnvSetting().DISABLE_TE_LINEAR_PATCHING
        ):
            _patch_te_linear_backward()
            self.fp8_backward_patched = True

    def switch_precision(self, new_selection: Union[str, LinearPrecision] = "original", reset_fp8_meta=True):
        # Switch preciesion: "original" or "fp8"
        selection = LinearPrecision(new_selection) if isinstance(new_selection, str) else new_selection
        if selection == self.selection:
            return
        if (
            selection == LinearPrecision.FP8
            and reset_fp8_meta
            and hasattr(self._precision_modules[LinearPrecision.FP8], "reset_fp8_meta_tensors")
        ):
            # reset_fp8_meta_tensors for fp8
            self._precision_modules[LinearPrecision.FP8].reset_fp8_meta_tensors()

        self.selection = selection

    def fp8_selection(self, if_select_fp8: bool):
        new_selection = "fp8" if if_select_fp8 else "original"
        self.switch_precision(new_selection)

    @property
    def use_input_current_scaling(self):
        return self.pre_cast_input_fp8_current_scaling

    @property
    def use_fp8(self):
        return self.selection == LinearPrecision.FP8

    def get_extra_state(self) -> torch.Tensor:
        fp8_extra_state = self._precision_modules[LinearPrecision.FP8].get_extra_state()
        state = {"selection": self.selection, "fp8_extra_state": fp8_extra_state}
        state_serialized = io.BytesIO()
        torch.save(state, state_serialized)
        return state_serialized

    def set_extra_state(self, state) -> None:
        assert isinstance(state, io.BytesIO)
        state.seek(0)
        state = torch.load(state)

        self.selection = state["selection"]
        self._precision_modules[LinearPrecision.FP8].set_extra_state(state["fp8_extra_state"])
