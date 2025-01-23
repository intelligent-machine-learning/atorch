import functools
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

from atorch.utils.version import torch_version

__all__ = ["get_total_norm", "clip_grads_with_norm_", "_no_grad"]


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


try:
    from torch.nn.utils import clip_grads_with_norm_, get_total_norm
except (ImportError, ModuleNotFoundError):
    if torch_version() >= (2, 5, 0):  # type: ignore
        from torch.utils._foreach_utils import (
            _device_has_foreach_support,
            _group_tensors_by_device_and_dtype,
            _has_foreach_support,
        )
    else:
        _device_has_foreach_support = None
        _group_tensors_by_device_and_dtype = None
        _has_foreach_support = None

    _tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

    @_no_grad
    def get_total_norm(
        tensors: _tensor_or_tensors,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""Compute the norm of an iterable of tensors.

        The norm is computed over the norms of the individual tensors, as if the norms of
        the individual tensors were concatenated into a single vector.

        Args:
            tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will be normalized
            norm_type (float): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
                Default: ``False``
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            Total norm of the tensors (viewed as a single vector).
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        else:
            tensors = list(tensors)
        norm_type = float(norm_type)
        if len(tensors) == 0:
            return torch.tensor(0.0)
        first_device = tensors[0].device
        grouped_tensors: Dict[
            Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
        ] = _group_tensors_by_device_and_dtype(
            [tensors]  # type: ignore[list-item]
        )  # type: ignore[assignment]

        norms: List[Tensor] = []
        for (device, _), ([device_tensors], _) in grouped_tensors.items():
            if (foreach is None and _has_foreach_support(device_tensors, device)) or (
                foreach and _device_has_foreach_support(device)
            ):
                norms.extend(torch._foreach_norm(device_tensors, norm_type))
            elif foreach:
                raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_tensors])

        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        return total_norm

    @_no_grad
    def clip_grads_with_norm_(
        parameters: _tensor_or_tensors,
        max_norm: float,
        total_norm: torch.Tensor,
        foreach: Optional[bool] = None,
    ) -> None:
        r"""Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

        The gradients will be scaled by the following calculation

        .. math::
            grad = grad * \frac{max\_norm}{total\_norm + 1e-6}

        Gradients are modified in-place.

        This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
        total norm.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float): max norm of the gradients
            total_norm (Tensor): total norm of the gradients to use for clipping
            foreach (bool): use the faster foreach-based implementation.
                If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
                fall back to the slow implementation for other device types.
                Default: ``None``

        Returns:
            None
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        if len(grads) == 0:
            return
        grouped_grads: Dict[
            Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
        ] = _group_tensors_by_device_and_dtype(
            [grads]
        )  # type: ignore[assignment]

        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for (device, _), ([device_grads], _) in grouped_grads.items():
            if (foreach is None and _has_foreach_support(device_grads, device)) or (
                foreach and _device_has_foreach_support(device)
            ):
                torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
            elif foreach:
                raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
            else:
                clip_coef_clamped_device = clip_coef_clamped.to(device)
                for g in device_grads:
                    g.mul_(clip_coef_clamped_device)
