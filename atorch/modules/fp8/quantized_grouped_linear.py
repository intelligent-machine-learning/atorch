import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from atorch.common.env import EnvSetting

try:
    import kitchen
    from kitchen import grouped_linear, quantization
    from kitchen.base import LinearBaseModule
    from kitchen.distributed import set_tensor_model_parallel_attributes
    from kitchen.linear import _QuantizedLinear
    from kitchen.utils import FP8GlobalStateManager, cast_if_needed, divide, init_method_constant, is_rank_0

    HAVE_KITCHEN = True
except (ImportError, ModuleNotFoundError):
    kitchen = None
    set_tensor_model_parallel_attributes = None
    FP8GlobalStateManager = None
    init_method_constant = None
    is_rank_0 = None
    cast_if_needed = None
    divide = None
    quantization = None
    _QuantizedLinear = object
    LinearBaseModule = object
    grouped_linear = None

    HAVE_KITCHEN = False


logger = logging.getLogger(__name__)


class _ATorchQuantizedGroupedLinear(_QuantizedLinear):
    """Optimized GroupedLinear Function with quantization support.

    The tensors x, w, bias, y, dy, wgrad, dgrad and bgrad should
    have the same dtype (FP32, BF16 or FP16). The GEMM output
    (y, wgrad, dgrad, bgrad) dtype could be further controlled
    by the qlinear_params.mm_(fprop|dgrad|wgrad).out_dtype.

    GEMM Equation in cuBLAS terms for TN layout

    Hopper FP8 GEMM only supports TN layout. So, the equations become:

    fwd   (TN):  Y = w @ x            # [n, k] x [m, k] = [m, n]
    dgrad (TN): dX = w.t() @ dY       # [k, n] x [m, n] = [m, k]
    wgrad (TN): dW = x.t() @ dY.t()   # [k, m] x [n, m] = [n, k]
    """

    @staticmethod
    def forward(  # type: ignore
        ctx,
        x: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        x_metas,
        weight_qresult_caches,
        qlinear_params,
        activation_dtype: torch.dtype,
        is_grad_enabled: bool,
        is_first_microbatch,
        fuse_wgrad_accumulation: bool,
        *weights_and_biases: torch.Tensor,
    ) -> torch.Tensor:

        # Get quantization op
        assert qlinear_params.quantize_op is not None
        quantize_op = qlinear_params.quantize_op

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = list(weights_and_biases[num_gemms:])
        device = x.device

        # Set dtypes, shapes and bias
        x_shape = x.shape
        out_dtype = x.dtype if qlinear_params.mm_fprop.out_dtype is None else qlinear_params.mm_fprop.out_dtype
        in_features = weights[0].shape[-1]
        assert x_shape[-1] == in_features, "GEMM not possible"
        x_list = torch.split(x.view(-1, in_features), m_splits)
        if use_bias:
            assert len(biases) == num_gemms, "Number of biases should match number of GEMMs."
            biases = [cast_if_needed(b, activation_dtype) for b in biases]

        # Quantize x (with optional transpose).
        qx_list, sx_list, qx_t_list, sx_t_list = [], [], [], []
        for x_i in x_list:
            # TODO: Add fused_multi_quantize
            qresult_x = quantize_op.quantize(
                x_i,
                qlinear_params.x_params,
                return_transpose=is_grad_enabled,
            )
            qx_list.append(qresult_x.data)
            sx_list.append(qresult_x.scale)
            if is_grad_enabled:
                qx_t_list.append(qresult_x.data_t)
                sx_t_list.append(qresult_x.scale_t)

        # Quantize w only for is_first_microbatch is True or None
        qw_list, sw_list, qw_t_list, sw_t_list = [], [], [], []
        update_fp8_weights = is_first_microbatch or is_first_microbatch is None
        if (
            EnvSetting().FORCE_QUANTIZE_PER_MICROBATCH
            or update_fp8_weights
            or (is_grad_enabled and weight_qresult_caches[0].data_t is None)
        ):
            # Quantize w (with optional transpose)
            # TODO: Add fused_multi_quantize
            for w, weight_qresult_cache in zip(weights, weight_qresult_caches):
                qresult_w = quantize_op.quantize(w, qlinear_params.w_params, return_transpose=is_grad_enabled)
                qw_list.append(qresult_w.data)
                sw_list.append(qresult_w.scale)
                if is_grad_enabled:
                    qw_t_list.append(qresult_w.data_t)
                    sw_t_list.append(qresult_w.scale_t)

                if not EnvSetting().FORCE_QUANTIZE_PER_MICROBATCH:
                    # save FP8 weight cache between microbatches
                    weight_qresult_cache.data = qresult_w.data
                    weight_qresult_cache.scale = qresult_w.scale
                    weight_qresult_cache.data_t = qresult_w.data_t
                    weight_qresult_cache.scale_t = qresult_w.scale_t
        else:
            for weight_qresult_cache in weight_qresult_caches:
                qresult_w = weight_qresult_cache.to_qresult()
                qw_list.append(qresult_w.data)
                sw_list.append(qresult_w.scale)
                if is_grad_enabled:
                    qw_t_list.append(qresult_w.data_t)
                    sw_t_list.append(qresult_w.scale_t)

        out = torch.empty(
            [sum(m_splits), weights[0].size(0)],
            dtype=out_dtype,
            device=device,
        )
        quantize_op.grouped_qgemm(
            qx_list,
            qw_list,
            qlinear_params.mm_fprop,
            sx_list,
            sw_list,
            list(torch.split(out, m_splits)),
            bias=biases if use_bias else None,
            qparams_x=qlinear_params.x_params,
            qparams_w=qlinear_params.w_params,
        )

        # Save context for backward pass
        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.use_bias = use_bias
            ctx.x_shape = x_shape
            ctx.qlinear_params = qlinear_params
            ctx.quantize_op = quantize_op
            ctx.is_first_microbatch = is_first_microbatch
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.device = device

            ctx.save_for_backward(
                *qx_t_list,
                *sx_t_list,
                *qw_t_list,
                *sw_t_list,
                *weights,
            )

        return out.view(-1, *x_shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, *dy_tup: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # Get saved tensors and sanity checks
        assert len(dy_tup) == 1
        dy = dy_tup[0]

        qx_t_list = ctx.saved_tensors[: ctx.num_gemms]
        sx_t_list = ctx.saved_tensors[ctx.num_gemms : 2 * ctx.num_gemms]
        qw_t_list = ctx.saved_tensors[2 * ctx.num_gemms : 3 * ctx.num_gemms]
        sw_t_list = ctx.saved_tensors[3 * ctx.num_gemms : 4 * ctx.num_gemms]
        weights = ctx.saved_tensors[4 * ctx.num_gemms : 5 * ctx.num_gemms]

        # Set dtypes, shapes
        out_dtype_dgrad = (
            dy.dtype if ctx.qlinear_params.mm_dgrad.out_dtype is None else ctx.qlinear_params.mm_dgrad.out_dtype
        )
        dy = dy.contiguous()
        dy = dy.view(-1, dy.shape[-1])
        dy_list = torch.split(dy, ctx.m_splits)

        # Bias gradient
        b_grad_list = []
        # Quantize dy (with transpose)
        qdy_list, sdy_list, qdy_t_list, sdy_t_list = [], [], [], []
        # TODO: Add fused_multi_quantize
        for dy_i in dy_list:
            b_grad_list.append(torch.sum(dy_i, dim=0) if ctx.use_bias else None)
            qresult_dy = ctx.quantize_op.quantize(dy_i, ctx.qlinear_params.g_params, return_transpose=True)
            qdy_list.append(qresult_dy.data)
            sdy_list.append(qresult_dy.scale)
            qdy_t_list.append(qresult_dy.data_t)
            sdy_t_list.append(qresult_dy.scale_t)

        # GEMM dgrad
        dgrad = torch.empty(
            (sum(ctx.m_splits), weights[0].size(1)),
            dtype=out_dtype_dgrad,
            device=ctx.device,
        )
        ctx.quantize_op.grouped_qgemm(
            qdy_list,
            qw_t_list,
            ctx.qlinear_params.mm_dgrad,
            sdy_list,
            sw_t_list,
            list(torch.split(dgrad, ctx.m_splits)),
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.w_params,
        )

        if ctx.is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
        else:
            accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

        out_dtype_wgrad = (
            dy.dtype if ctx.qlinear_params.mm_wgrad.out_dtype is None else ctx.qlinear_params.mm_wgrad.out_dtype
        )

        wgrad_list = [
            (
                w.main_grad
                if ctx.fuse_wgrad_accumulation
                else torch.empty(w.size(), dtype=out_dtype_wgrad, device=ctx.device)
            )
            for w in weights
        ]
        # GEMM wgrad
        ctx.quantize_op.grouped_qgemm(
            qdy_t_list,
            qx_t_list,
            ctx.qlinear_params.mm_wgrad,
            sdy_t_list,
            sx_t_list,
            wgrad_list,
            accumulate=accumulate_wgrad_into_param_main_grad,
            qparams_x=ctx.qlinear_params.g_params,
            qparams_w=ctx.qlinear_params.x_params,
        )
        # Handle wgrad accumulation for mcore if needed
        wgrad_list = [_QuantizedLinear.reset_wgrad_if_needed(ctx, w, wgrad) for w, wgrad in zip(weights, wgrad_list)]

        return (
            dgrad.view(ctx.x_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *wgrad_list,
            *b_grad_list,
        )


class ATorchGroupedLinear(LinearBaseModule):
    """Applies linear transformations to the incoming data list
       :math:`y_i = x_iA_i^T + b_i` in a grouped way.
       For better performance, the input and output tensors are expected to be one single tensor.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simutaneously.
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.
    ub_overlap_rs : bool, default = `False`
                    Reduce-Scatter overlap with GEMM by pipelining the GEMM and Reduce-Scatter.
    ub_overlap_ag : bool, default = `False`
                    All-Gather overlap with GEMM by pipelining the GEMM and All-Gather.

    Note: GroupedLinear doesn't really handle the TP communications inside. The `tp_size` and
          `parallel_mode` are used to determine the shapes of weights and biases.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.

    Quantization parameters
    -----------------------
    qlinear_params : Optional[config.QLinearParams], default = `None`
                     used to set quantization linear parameters for the linear layer.
                     If not provided, currently it will be determined from ENV Variable `QAT_PARAMS`.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        layer_number: Optional[int] = None,
        qlinear_params=None,
    ) -> None:
        super().__init__(tp_group, tp_size, sequence_parallel)

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinear doesn't support bias when TP > 1. "
                "Because the TP communication is handled outside of this module."
            )
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        assert not ub_overlap_rs and not ub_overlap_ag, "GroupedLinear doesn't support Userbuffer overlap."
        self.layer_name = ub_name
        self.layer_number = layer_number
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        # Set quantization parameters
        self.qlinear_params = self.set_qlinear_params(
            qlinear_params=qlinear_params,
            layer_number=self.layer_number,
            layer_name=self.layer_name,
        )

        if is_rank_0():
            logger.info(
                f"local_qlinear_params for layer {layer_number} grouped linear module"
                f" {self.layer_name} is set to {self.qlinear_params}"
            )

        self.parallel_mode = parallel_mode
        assert self.parallel_mode in (
            "row",
            "column",
            None,
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.is_fp8_params_enabled()
        # self.fp8 = FP8GlobalStateManager.is_fp8_enabled()
        # assert self.fp8, "FP8 must be enabled for Linear module"
        assert with_fp8_params is False, "FP8 only parameters are not supported yet"
        assert self.fsdp_group is None, "FSDP sharding is not supported yet"

        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                setattr(
                    self,
                    f"bias{i}",
                    torch.Tensor().to(dtype=params_dtype, device=device),
                )

        self.reset_parameters(defer_init=(device == "meta"))

        # quantized weight cache between microbatches
        self.weight_qresult_caches = [
            quantization.QuantizeResultCache(data=None, scale=None, data_t=None, scale_t=None)
            for _ in range(self.num_gemms)
        ]

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for i in range(self.num_gemms):
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, f"bias{i}"),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

    @torch._dynamo.disable(recursive=True)
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
        inp_metas=None,
    ) -> Tuple[Any, List[torch.Tensor]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
              Input tensor.
        m_splits : List[int]
                   List of integers representing the split of the input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        self.set_activation_dtype(inp)
        inp = inp.contiguous()
        # Sanity checks
        assert self.tp_group_initialized, "TP group not initialized"
        input_dtypes = (torch.float, torch.float16, torch.bfloat16, torch.float32)
        assert inp.dtype in input_dtypes, f"Input dtype {inp.dtype} not supported"
        assert all(
            isinstance(weight_qresult_cache, quantization.QuantizeResultCache)
            for weight_qresult_cache in self.weight_qresult_caches
        ), "weight_qresult_cache must be QuantizeResultCache"
        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."
        assert self.qlinear_params.quantize, "Quantization must be enabled for GroupedLinear"

        weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

        if torch.is_grad_enabled():
            linear_fn = _ATorchQuantizedGroupedLinear.apply
            args: List[Any] = []
        else:
            linear_fn = _ATorchQuantizedGroupedLinear.forward
            args = [None]
        args += (
            inp,
            m_splits,
            self.apply_bias,
            inp_metas,
            self.weight_qresult_caches,
            self.qlinear_params,
            self.activation_dtype,
            torch.is_grad_enabled(),
            is_first_microbatch,
            self.fuse_wgrad_accumulation,
            *weight_tensors,
            *bias_tensors,
        )
        out = linear_fn(*args)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out


def patch_kitchen_grouped_linear():
    if kitchen is None or grouped_linear is None:
        return

    grouped_linear._QuantizedGroupedLinear = _ATorchQuantizedGroupedLinear
    print("Patch kitchen quantized grouped linear")

    grouped_linear.GroupedLinear = ATorchGroupedLinear
    print("Patch kitchen grouped linear")


if HAVE_KITCHEN:
    patch_kitchen_grouped_linear()
