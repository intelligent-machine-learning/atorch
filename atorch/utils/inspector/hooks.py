import os
import re

import numpy as np
import torch

import atorch

try:
    import transformer_engine.pytorch.cpp_extensions as texcpp

    try:
        import transformer_engine_torch as tex
    except (ImportError, ModuleNotFoundError):
        import transformer_engine_extensions as tex  # te version < 1.12
    from transformer_engine.common.recipe import Format
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.numerics_debug import fp8_tensor_statistics

    from atorch.modules.fp8 import PrecisionSwitchableLinear

    _te_available = True
    _te_cast_fp8_available = hasattr(tex, "cast_to_fp8") and hasattr(texcpp, "cast_to_fp8")
except (ImportError, ModuleNotFoundError):
    _te_available = False
    _te_cast_fp8_available = False

from atorch.common.log_utils import default_logger as logger
from atorch.modules.fp8 import ScaledLinear
from atorch.modules.fp8.quantize import get_fp8_quantize_underflows

try:
    import matplotlib.pyplot as plt
except (ImportError, ModuleNotFoundError):
    plt = None


"""This inspector feature implementation is taken from Nvidia's inspector tool for Megatron with
   modification to work with non-te ops and non-Megatron models. """


class TensorInspector:
    def __init__(
        self,
        log_tensor_interval,
        curr_iteration=0,
        save_tensor=False,
        save_tensor_dir="tensor_dir",
        plot_tensor=False,
        plot_tensor_dir="plot_dir",
        rank=None,
        log_fn=None,
        enable=True,
        summary_writer=None,
        summary_writer_items=None,
        log_underflows_for_linear=False,
    ):
        self.log_tensor_interval = log_tensor_interval
        self.curr_iteration = curr_iteration
        self.save_tensor = save_tensor
        self.save_tensor_dir = save_tensor_dir
        self.plot_tensor = plot_tensor
        self.plot_tensor_dir = plot_tensor_dir
        self.rank = rank if rank is not None else atorch.rank()
        self.log_fn = log_fn if log_fn is not None else lambda string: print(f"[Rank {self.rank}] {string}")
        self.enable = enable
        # If summary_writer is not None, add inspector results into tensorboard using summary_writer
        self.summary_writer = summary_writer
        # If summary_writer_items is None, write all items into summary_writer. If not None, it is a list of item names.
        # Only these items are written by summary_writer.
        # Valid names are: "amin", "amax", "current_amax_scale",
        # and "fp8_meta_scale", "fp8_cos_similarity", "fp8_mse", "fp8_underflows", "fp8_overflows"
        self.log_underflows_for_linear = log_underflows_for_linear  # if log underflows for non-fp8 linear
        self.summary_writer_items = summary_writer_items
        if save_tensor and not os.path.exists(save_tensor_dir):
            try:
                os.makedirs(save_tensor_dir)
            except OSError:
                logger.error(f"Cannot create directory {save_tensor_dir}, save_tensor is disabled.")
                self.save_tensor = False
        if plot_tensor and not os.path.exists(plot_tensor_dir):
            try:
                os.makedirs(plot_tensor_dir)
            except OSError:
                logger.error(f"Cannot create directory {plot_tensor_dir}, plot_tensor is disabled.")
                self.plot_tensor = False

    def enable(self, if_enable):
        self.enable = if_enable

    def step(self, cur_iter=None):
        if cur_iter is not None:
            self.curr_iteration = cur_iter
        else:
            self.curr_iteration += 1

    def register_hooks(
        self,
        model,
        log_tensor_name_pattern="(qkv|proj|fc)",
        exclude_tensor_name_pattern=None,
        layer_types=(torch.nn.Linear,),
        backward_use_e4m3=False,
    ):
        """Register log tensor hook and save tensor hook"""

        matched_modules = []
        for name, layer in model.named_modules():
            # Remove checkpoint and fsdp1 wrapper name.
            name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
            if (
                re.search(log_tensor_name_pattern, name)
                and (exclude_tensor_name_pattern is None or not re.search(exclude_tensor_name_pattern, name))
                and (
                    (_te_available and isinstance(layer, (TransformerEngineBaseModule, PrecisionSwitchableLinear)))
                    or isinstance(layer, ScaledLinear)
                    or (layer_types is not None and isinstance(layer, layer_types))
                )
            ):

                # log tensor hook
                layer.register_forward_hook(log_tensor_hook(name, self, is_fwd=True))
                layer.register_full_backward_hook(
                    log_tensor_hook(name, self, is_fwd=False, backward_use_e4m3=backward_use_e4m3)
                )

                # save tensor hook
                if self.save_tensor:
                    layer.register_forward_hook(save_tensor_hook(name, self, is_fwd=True))
                    layer.register_full_backward_hook(save_tensor_hook(name, self, is_fwd=False))

                # plot tensor hook
                if self.plot_tensor:
                    layer.register_forward_hook(plot_tensor_hook(name, self, is_fwd=True))
                    layer.register_full_backward_hook(plot_tensor_hook(name, self, is_fwd=False))

                matched_modules.append(name)

        return matched_modules


def calculate_percentiles(activations):
    min_vals = np.min(activations, axis=0)
    max_vals = np.max(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)
    return min_vals, max_vals, p1, p99, p25, p75


def plot_activation_distribution(tensor, save_image_path, save_image_name="Tensor Value Distribution"):
    """
    Plot the distribution of activations from a PyTorch tensor and save the plot as an image.

    Parameters:
    - tensor_path (str): Path to the PyTorch tensor file.
    - save_image_path (str): Path to save the generated image.
    - save_image_name (str): Title of the plot.
    """

    if plt is None:
        logger.error("Need to install matplotlib to support plot.")
        return

    # Calculate percentiles and mean/std for the activations
    minv, maxv, p1, p99, p25, p75 = calculate_percentiles(tensor)

    # Create a figure and axis
    fig, axs = plt.subplots(1, 1, figsize=(7, 6))

    # Generate the hidden dimension index
    hidden_dimension_index = np.arange(tensor.shape[1])

    # Plot the distribution of activations
    axs.fill_between(hidden_dimension_index, minv, maxv, color="red", alpha=0.2, label="Min/Max")
    axs.fill_between(hidden_dimension_index, p1, p99, color="blue", alpha=0.3, label="1/99 Percentile")
    axs.fill_between(hidden_dimension_index, p25, p75, color="orange", alpha=0.5, label="25/75 Percentile")

    axs.set_title(f"{save_image_name}")
    axs.set_xlabel("Hidden dimension index")
    axs.set_ylabel("Value")

    # Add legend
    axs.legend(loc="upper right")

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"Save figure to file: {save_image_path}")
    plt.savefig(save_image_path)


def qdq(inp, meta, fp8_tensor, fp8_format="e4m3", use_current_scaling=False, amax=None, scale=None):
    """Q and DQ tensor"""
    fp8_type = tex.DType.kFloat8E4M3 if fp8_format == "e4m3" else tex.DType.kFloat8E5M2

    if use_current_scaling:
        scale_inv = 1.0 / scale
        ret = tex.cast_to_fp8(inp, scale, amax, scale_inv, fp8_type)
        if inp.dtype == torch.bfloat16:
            odtype = tex.DType.kBFloat16
        elif inp.dtype == torch.float16:
            odtype = tex.DType.kFloat16
        else:
            assert inp.dtype == torch.float32
            odtype = tex.DType.kFloat32
        ret = tex.cast_from_fp8(ret, scale_inv, fp8_type, odtype)
    else:
        ret = texcpp.cast_to_fp8(inp, meta, fp8_tensor, fp8_type)
        ret = texcpp.cast_from_fp8(ret, meta, fp8_tensor, fp8_type, TE_DType[inp.dtype])

    return ret


def cosine(tensor1, tensor2):
    cosine_sim = torch.nn.functional.cosine_similarity(tensor1.reshape(-1), tensor2.reshape(-1), dim=0).item()
    return cosine_sim


def get_fp8_meta(m):
    input_current_scaling = False
    if not _te_available:
        return None, False
    if isinstance(m, PrecisionSwitchableLinear):
        input_current_scaling = m.use_input_current_scaling
        m = m._precision_modules[m.selection]

    if hasattr(m, "fp8") and m.fp8:
        return m.fp8_meta, input_current_scaling
    return None, False


def save_tensor_hook(module_name, inspector, is_fwd=True):
    """Set up hook for forward or backpropagation"""
    save_dir = inspector.save_tensor_dir
    rank = inspector.rank
    interval = inspector.log_tensor_interval
    log_fn = inspector.log_fn

    if is_fwd:
        tensor_names = ["fwd_x", "fwd_y"]
        fp8_meta_key = "scaling_fwd"
        fp8_gemm_type = tex.FP8FwdTensors.GEMM1_INPUT if _te_available else None
    else:
        tensor_names = ["bwd_dx", "bwd_dy"]
        fp8_meta_key = "scaling_bwd"
        fp8_gemm_type = tex.FP8BwdTensors.GRAD_OUTPUT1 if _te_available else None

    def hook(module, inputs, outputs):
        """Save input and output tensor to file"""
        step = inspector.curr_iteration
        # skip step == 0
        if inspector.enable and ((step) % interval) == 0 and step != 0:
            # when backward, bwd_dx may be None if x does not require grad.
            if inputs[0] is not None:
                tensors = {
                    tensor_names[0]: inputs[0].detach().cpu(),  # x or dx
                }
            else:
                tensors = {}

            tensors[tensor_names[1]] = outputs[0].detach().cpu()  # y or dy

            # save weight
            if is_fwd:
                params = list(module.parameters())
                if len(params) > 0:
                    tensors["fwd_w"] = params[0].detach().cpu()
                elif hasattr(module, "weight"):
                    tensors["fwd_w"] = module.weight.detach().cpu()

            # save tensor and scale_inv (if fp8) to file
            for tensor_name, tensor in tensors.items():
                filename = os.path.join(save_dir, f"{module_name}.{tensor_name}.step{step}.rank{rank:03d}.pt")
                if not os.path.exists(filename):
                    # TODO: only save the first micro batch, for now
                    save_obj = {"tensor": tensor}
                    fp8_meta, _ = get_fp8_meta(module)
                    if fp8_meta is not None and (tensor_name in ("fwd_x", "bwd_dy")):
                        save_obj["scale"] = fp8_meta[fp8_meta_key].scale[fp8_gemm_type]
                    torch.save(save_obj, filename)

                    #           log_fn(f'[save tensor hook] name: {tensor_name}, shape: {tensor.shape}')
                    log_fn(
                        f"[save tensor hook] step: {step}, "
                        f"layer: {module_name}, save {tensors.keys()} to dir: {save_dir}"
                    )

    return hook


def log_tensor_hook(module_name, inspector, is_fwd=True, backward_use_e4m3=False):
    """Set up hook for forward or backpropagation"""
    interval = inspector.log_tensor_interval
    log_fn = inspector.log_fn

    if is_fwd:
        tensor_name = ["fwd_x", "fwd_w"]
        fp8_meta_key = "scaling_fwd"
        fp8_gemm_type = [tex.FP8FwdTensors.GEMM1_INPUT, tex.FP8FwdTensors.GEMM1_WEIGHT] if _te_available else None
        fp8_fmt = "e4m3"
    else:
        tensor_name = [
            "bwd_dy",
        ]
        fp8_meta_key = "scaling_bwd"
        fp8_gemm_type = (
            [
                tex.FP8BwdTensors.GRAD_OUTPUT1,
            ]
            if _te_available
            else None
        )
        fp8_fmt = "e4m3" if backward_use_e4m3 else "e5m2"

    def hook(module, inputs, outputs):

        step = inspector.curr_iteration
        # skip step == 0
        if inspector.enable and (step % interval) == 0 and step != 0:

            # get target tensor: (fwd_x & fwd_w) or bwd_dy
            targets = [
                inputs[0] if is_fwd else outputs[0],
            ]
            if is_fwd:
                for p_name, p_tensor in module.named_parameters(recurse=False):
                    if p_name == "weight":
                        targets.append(p_tensor)
                if len(targets) == 1 and hasattr(module, "weight"):
                    targets.append(module.weight)

            # process each target tensor
            for index, tensor in enumerate(targets):
                tb_dict = {}
                tensor = tensor.detach()
                # Remove rows of all zeros (in SFT, we see lots of rows of zero in bwd_dy)
                # tensor = remove_zero_rows(tensor.detach())

                # nonzero abs min and max
                nonzeros = tensor[tensor.nonzero(as_tuple=True)]
                # nonzeros = tensor

                amin, amax = torch.abs(nonzeros).aminmax()
                # scale from current amax
                act_scale = (
                    (Format.HYBRID.value.max_fwd if is_fwd else Format.HYBRID.value.max_bwd) / amax
                    if _te_available
                    else (448 / amax if is_fwd else 57344 / amax)
                )

                log_str = (
                    f"{module_name}.{tensor_name[index]}.step{step:d}"
                    f", amin: {amin:.3e}"
                    f", amax: {amax:.3e}"
                    f", amax scale: {act_scale:.3e}"
                )

                tb_dict["amin"] = amin
                tb_dict["amax"] = amax
                tb_dict["current_amax_scale"] = act_scale

                # FP8 quantization error
                fp8_meta, input_current_scaling = get_fp8_meta(module)
                use_current_scaling = is_fwd and index == 0 and input_current_scaling
                if fp8_meta is not None and _te_cast_fp8_available:
                    # Q then DQ
                    t_fp8 = qdq(
                        nonzeros,
                        fp8_meta[fp8_meta_key],
                        fp8_gemm_type[index],
                        fp8_fmt,
                        use_current_scaling,
                        amax,
                        act_scale,
                    )

                    # percentage of underflows and overflows
                    # for underflows, exclude those existing zeros
                    numel = torch.numel(tensor)
                    num_zeros_fp8, num_max_fp8 = fp8_tensor_statistics(t_fp8, fp8_fmt)
                    pct_underflows = num_zeros_fp8 * 100.0 / numel
                    pct_overflows = num_max_fp8 * 100.0 / numel

                    # cos and mse from fp8 quantization
                    cos = cosine(nonzeros, t_fp8)
                    mse = torch.nn.functional.mse_loss(nonzeros, t_fp8)

                    # scale from meta
                    scale = fp8_meta[fp8_meta_key].scale[fp8_gemm_type[index]]

                    log_str += (
                        f", meta scale: {scale:.3e}"
                        f", cos: {cos:.3f}"
                        f", mse: {mse:.3e}"
                        f", underflows(%): {pct_underflows:.1f}"
                        f", overflows(%): {pct_overflows:.1f}"
                    )
                    tb_dict["fp8_meta_scale"] = scale
                    tb_dict["fp8_cos_similarity"] = cos
                    tb_dict["fp8_mse"] = mse
                    tb_dict["fp8_underflows"] = pct_underflows
                    tb_dict["fp8_overflows"] = pct_overflows
                if isinstance(module, ScaledLinear) or inspector.log_underflows_for_linear:
                    fp8_format = torch.float8_e4m3fn if fp8_fmt == "e4m3" else torch.float8_e5m2
                    results = get_fp8_quantize_underflows(tensor.contiguous(), fp8_format)
                    log_str += (
                        f", tensorwise_underflows(%): {results['tensorwise']:.1f}"
                        f", rowwise_underflows(%): {results['rowwise']:.1f}"
                        f", colwise_underflows(%): {results['colwise']:.1f}"
                    )
                    tb_dict["fp8_tensorwise_underflows"] = results["tensorwise"]
                    tb_dict["fp8_rowwise_underflows"] = results["rowwise"]
                    tb_dict["fp8_colwise_underflows"] = results["colwise"]
                    if "tilewise" in results:
                        log_str += f", tilewise_underflows(%): {results['tilewise']:.1f}"
                        tb_dict["fp8_tilewise_underflows"] = results["tilewise"]
                    if "blockwise" in results:
                        log_str += f", blockwise_underflows(%): {results['blockwise']:.1f}"
                        tb_dict["fp8_blockwise_underflows"] = results["blockwise"]

                log_fn(log_str)
                if inspector.summary_writer is not None:
                    # write to tensorboard
                    tname = "TensorInspect/"
                    if inspector.rank is not None:
                        tname = tname + f"rank{inspector.rank}/"
                    tname += f"{module_name}.{tensor_name[index]}"
                    for item_name, item_value in tb_dict.items():
                        if inspector.summary_writer_items is None or item_name in inspector.summary_writer_items:
                            inspector.summary_writer.add_scalar(f"{tname}/{item_name}", item_value, step)

    return hook


def plot_tensor_hook(module_name, inspector, is_fwd=True):
    """Set up hook for forward or backpropagation"""
    save_dir = inspector.plot_tensor_dir
    rank = inspector.rank
    interval = inspector.log_tensor_interval
    log_fn = inspector.log_fn

    tensor_name = (
        [
            "fwd_x",
        ]
        if is_fwd
        else [
            "bwd_dy",
        ]
    )

    def hook(module, inputs, outputs):
        step = inspector.curr_iteration
        # skip step == 0
        if inspector.enable and (step % interval) == 0 and step != 0:
            # get target tensor: (fwd_x & fwd_w) or bwd_dy
            targets = [
                inputs[0] if is_fwd else outputs[0],
            ]

            # TODO: enable weight later on
            """
        if is_fwd:
          for p_name, p_tensor in module.named_parameters(recurse=False):
            if p_name == "weight":
              targets.append(p_tensor)
        """

            # process each target tensor
            for index, tensor in enumerate(targets):
                # Get tensor as numpy
                tensor = tensor.detach().cpu().float().numpy()
                last_dim = tensor.shape[-1]
                tensor = tensor.reshape(-1, last_dim)
                # Plot and save figure
                filename = os.path.join(save_dir, f"{module_name}.{tensor_name[index]}.step{step}.rank{rank:03d}.png")
                plot_activation_distribution(tensor, filename)
                log_fn(f"[plot tensor hook] Save tensor plot figure: {filename}")

    return hook
