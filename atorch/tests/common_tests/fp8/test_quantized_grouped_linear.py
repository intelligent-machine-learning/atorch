# flake8: noqa: E402
import pytest
import torch

pytestmark = pytest.mark.fp8

try:
    import transformer_engine as te

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

from atorch.utils.version import torch_version

if torch_version() >= (2, 4, 0):  # type: ignore
    from atorch.modules.fp8.quantized_grouped_linear import HAVE_KITCHEN
else:
    HAVE_KITCHEN = False

if HAVE_KITCHEN:
    import kitchen
    from kitchen import (
        ops,
        quantization,
        quantization_per_channel,
        quantization_per_channel_ref,
        quantization_per_tensor,
        quantization_per_tensor_ref,
        quantization_subchannel_block_hybrid,
        quantization_subchannel_block_hybrid_ref,
    )
    from kitchen.config import QLinearParams
    from kitchen.distributed import allreduce
    from kitchen.grouped_linear import GroupedLinear
    from kitchen.linear import Linear
    from kitchen.quantization import MMParams, QParams, ScalingType

    per_channel_gemm_supported = kitchen.ops.gemm.is_per_channel_gemm_supported()
else:
    kitchen = None
    ops = None
    quantization = None
    quantization_per_channel = None
    quantization_per_channel_ref = None
    quantization_per_tensor = None
    quantization_per_tensor_ref = None
    quantization_subchannel_block_hybrid = None
    quantization_subchannel_block_hybrid_ref = None
    allreduce = None
    QLinearParams = None
    Linear = None
    GroupedLinear = None
    QParams, ScalingType, MMParams = None, None, None
    per_channel_gemm_supported = None


class Recipe:
    @staticmethod
    def none():
        return None

    @staticmethod
    def predefined_fp8_per_tensor():
        _scaling_type = ScalingType.PER_TENSOR
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            mm_fprop=MMParams(use_split_accumulator=False),
            quantize_op=quantization_per_tensor.QuantizeOpFP8PerTensor(),
            allgather_fp8=True,
        )

    @staticmethod
    def nonquantize_linear():
        return QLinearParams(
            quantize=False,
        )

    @staticmethod
    def torch_nn_linear():
        return None

    @staticmethod
    def quantize_linear_nonquantize():
        return QLinearParams(
            quantize_op=quantization.QuantizeOpNonQuantize(),
        )

    @staticmethod
    def fp8_per_tensor():
        _scaling_type = ScalingType.PER_TENSOR
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            quantize_op=quantization_per_tensor.QuantizeOpFP8PerTensor(),
        )

    @staticmethod
    def fp8_per_tensor_with_norm_fusion():
        _scaling_type = ScalingType.PER_TENSOR
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            quantize_op=quantization_per_tensor.QuantizeOpFP8PerTensor(),
            norm_fusion=True,
        )

    @staticmethod
    def fp8_per_tensor_ref():
        _scaling_type = ScalingType.PER_TENSOR
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            quantize_op=quantization_per_tensor_ref.QuantizeOpFP8PerTensorRef(),
        )

    @staticmethod
    def fp8_per_tensor_qgather(allgather_fp8: bool = False):
        _scaling_type = ScalingType.PER_TENSOR
        return QLinearParams(
            mm_fprop=MMParams(use_split_accumulator=False),
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            allgather_fp8=allgather_fp8,
            quantize_op=quantization_per_tensor.QuantizeOpFP8PerTensor(),
        )

    @staticmethod
    def fp8_per_channel():
        _scaling_type = ScalingType.PER_CHANNEL
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            quantize_op=quantization_per_channel.QuantizeOpFP8PerChannel(),
        )

    @staticmethod
    def fp8_per_channel_ref():
        _scaling_type = ScalingType.PER_CHANNEL
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e5m2, scaling_type=_scaling_type),
            quantize_op=quantization_per_channel_ref.QuantizeOpFP8PerChannelRef(),
        )

    @staticmethod
    def fp8_per_channel_all_e4m3():
        _scaling_type = ScalingType.PER_CHANNEL
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type),
            quantize_op=quantization_per_channel.QuantizeOpFP8PerChannel(),
        )

    @staticmethod
    def fp8_per_sub_channel_cutlass():
        # fmt: off
        _scaling_type = ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(128, 128)),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            quantize_op=quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(ops.Backend.CUTLASS),
            allgather_fp8=False,
        )
        # fmt: on

    @staticmethod
    def fp8_per_sub_channel_cutlass_ref():
        # fmt: off
        _scaling_type = ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(128, 128)),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            quantize_op=quantization_subchannel_block_hybrid_ref.HybridBlockAndVectorTiledQuantizeOpRef(
                ops.Backend.CUTLASS
            ),
            allgather_fp8=False,
        )
        # fmt: on

    @staticmethod
    def fp8_per_sub_channel_cublas():
        # fmt: off
        _scaling_type = ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(128, 128)),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            quantize_op=quantization_subchannel_block_hybrid.HybridBlockAndVectorTiledQuantizeOp(ops.Backend.CUBLAS),
            allgather_fp8=False,
        )
        # fmt: on

    @staticmethod
    def fp8_per_sub_channel_cublas_ref():
        # fmt: off
        _scaling_type = ScalingType.VECTOR_TILED_X_AND_G_BLOCK_TILED_W
        return QLinearParams(
            x_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            w_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(128, 128)),
            g_params=QParams(quant_dtype=torch.float8_e4m3fn, scaling_type=_scaling_type, quant_tile_shape=(1, 128)),
            quantize_op=quantization_subchannel_block_hybrid_ref.HybridBlockAndVectorTiledQuantizeOpRef(
                ops.Backend.CUBLAS
            ),
            allgather_fp8=False,
        )
        # fmt: on


class TestLinearBase:
    @staticmethod
    def _prepare_data(batch_size, hidden_size, out_size, use_bias=True, seed=0, dtype=torch.float32):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
        w = torch.randn((out_size, hidden_size), dtype=dtype, device="cuda")
        bias = torch.randn((out_size), dtype=dtype, device="cuda") if use_bias else None
        gradient = torch.randn((batch_size, out_size), dtype=dtype, device="cuda")

        return x, w, bias, gradient

    @staticmethod
    def _shard_tensor(x, world_size, axis):
        split_size = x.size()[axis] // world_size
        split_tensor = torch.split(x, split_size, axis)
        out = []
        for tensor in split_tensor:
            out.append(tensor.detach().clone().requires_grad_(x.requires_grad))
        return out

    @staticmethod
    def _gather_tensor(local, world_size, tp_group, concat_dim):
        out_list = [torch.zeros_like(local) for _ in range(world_size)]
        torch.distributed.all_gather(out_list, local, tp_group)
        return torch.cat(out_list, dim=concat_dim)

    @classmethod
    def run_linear_one_step(cls, layer, x, gradient, is_first_microbatch=None, fuse_wgrad_accumulation=False):
        # reset gradients
        layer.zero_grad()
        x.grad = None

        # Forward pass
        if isinstance(layer, Linear):
            # Kitchen Linear
            y_q = layer.forward(x, is_first_microbatch=is_first_microbatch)
        else:
            # the default torch.nn.Linear
            y_q = layer(x)

        # Backward pass
        y_q.backward(gradient)

        # Collect gradients
        dgrad = x.grad
        bgrad = layer._parameters["bias"].grad if layer._parameters.get("bias", None) is not None else None
        assert "weight" in layer._parameters
        if fuse_wgrad_accumulation:
            wgrad = layer._parameters["weight"].main_grad
            assert layer._parameters["weight"].grad is None
        else:
            wgrad = layer._parameters["weight"].grad

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_multiple_steps(
        cls,
        layer,
        x,
        gradient,
        run_num_steps,
        enable_weight_cache,
        fuse_wgrad_accumulation=False,
    ):
        """
        Run multiple steps of linear layer and collect results.
        """

        y_q_list, dgrad_list, wgrad_list = [], [], []
        bgrad_list = [] if layer._parameters.get("bias", None) is not None else None

        for i in range(run_num_steps):
            x_i = (x + i).clone().detach().requires_grad_(True)
            # run_linear_one_step
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(
                layer,
                x_i,
                gradient,
                is_first_microbatch=(i == 0) if enable_weight_cache else None,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )

            # Collect results
            y_q_list.append(y_q.detach().clone())
            dgrad_list.append(dgrad.detach().clone())
            wgrad_list.append(wgrad.detach().clone())
            if bgrad_list is not None and bgrad is not None:
                bgrad_list.append(bgrad.detach().clone())

        # Stack the results
        return (
            torch.stack(y_q_list),
            torch.stack(dgrad_list),
            torch.stack(wgrad_list),
            torch.stack(bgrad_list) if bgrad_list is not None else None,
        )

    @classmethod
    def run_linear_preprocess_parallel(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_size=1,
        rank=0,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # split w in N dim, which should be axis 0
                w = cls._shard_tensor(w, tp_size, 0)[rank]
                bias = cls._shard_tensor(bias, tp_size, 0)[rank] if bias is not None else None
                # split gradient in N dim, which should be axis 1
                gradient = cls._shard_tensor(gradient, tp_size, 1)[rank]
                if sequence_parallel:
                    # split x in M dim, which should be axis 0
                    x = cls._shard_tensor(x, tp_size, 0)[rank]
            # row parallel, split x in k dim, which should be axis 1, split w in k dim, should be axis 1
            if parallel_mode == "row":
                # split x in K dim, which should be axis 1
                x = cls._shard_tensor(x, tp_size, 1)[rank]
                # split w in K dim, which should be axis 1
                w = cls._shard_tensor(w, tp_size, 1)[rank]
                if sequence_parallel:
                    # split gradient in M dim, which should be axis 0
                    gradient = cls._shard_tensor(gradient, tp_size, 0)[rank]
        return x, w, bias, gradient

    @classmethod
    def run_linear_postprocess_parallel(
        cls,
        y_q,
        dgrad,
        wgrad,
        bgrad,
        parallel_mode,
        sequence_parallel,
        tp_size,
        tp_group,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # gather y_q in N dim, which should be axis 1
                y_q = cls._gather_tensor(y_q, tp_size, tp_group, 1)
                # gather wgrad in N dim, which should be axis 0
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 0)
                # gather bgrad in N dim, which should be axis 0
                bgrad = cls._gather_tensor(bgrad, tp_size, tp_group, 0) if bgrad is not None else None
                if sequence_parallel:
                    # gather dgrad in M dim, which should be axis 0
                    dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 0)
            if parallel_mode == "row":
                # gather dgrad in K dim, which should be axis 1
                dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 1)
                # gather wgrad in K dim, which should be axis 1
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 1)
                if sequence_parallel:
                    # gather y_q in M dim, which should be axis 0
                    y_q = cls._gather_tensor(y_q, tp_size, tp_group, 0)
                    # we need to sum bias gradient when using TP + SP
                    bgrad, _ = allreduce(bgrad, tp_group) if bgrad is not None else None

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear(
        cls,
        x,
        w,
        bias,
        gradient,
        get_recipe,
        parallel_mode=None,
        sequence_parallel=False,
        tp_group=None,
        tp_size=1,
        rank=0,
        run_num_steps=1,
        enable_weight_cache=False,
        fuse_wgrad_accumulation=False,
    ):
        """
        If Model parallel, split inputs for a given rank and return the gathered output and gradients,
        so that they can be compared with the reference single GPU run.
        """
        # clone inputs and move to current device
        # w has shape [N, K], x has shape [M, K], gradient has shape [M, N]
        x = x.clone().detach().requires_grad_(True).to("cuda")
        w = w.clone().detach().to("cuda")
        gradient = gradient.clone().detach().to("cuda")
        bias = bias.clone().detach().to("cuda") if bias is not None else None
        in_features = x.shape[1]
        out_features = w.shape[0]

        # If Model parallel: split inputs for a given rank
        x, w, bias, gradient = cls.run_linear_preprocess_parallel(
            x, w, bias, gradient, parallel_mode, sequence_parallel, tp_size, rank
        )

        # set data types
        params_dtype = x.dtype

        # get quantization recipe
        qlinear_params = get_recipe()
        # Create linear layer and copy weights
        if get_recipe is Recipe.torch_nn_linear:
            layer = torch.nn.Linear(in_features, out_features, bias=bias is not None, dtype=params_dtype)
            assert not fuse_wgrad_accumulation
            assert tp_size == 1
        else:
            layer = Linear(
                in_features,
                out_features,
                bias=bias is not None,
                params_dtype=params_dtype,
                parallel_mode=parallel_mode,
                sequence_parallel=sequence_parallel,
                tp_group=tp_group,
                tp_size=tp_size,
                qlinear_params=qlinear_params,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )

        layer = layer.to("cuda")

        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:
                layer.bias.copy_(bias)

        if fuse_wgrad_accumulation:
            assert run_num_steps > 1, "Fused weight gradient accumulation requires run_num_steps > 1"
            layer.weight.main_grad = torch.zeros_like(layer.weight)

        # Run one step or multiple steps
        if run_num_steps == 1:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(layer, x, gradient)
        else:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_multiple_steps(
                layer,
                x,
                gradient,
                run_num_steps,
                enable_weight_cache,
                fuse_wgrad_accumulation,
            )

        # If Model parallel: gather output and gradients from all ranks
        y_q, dgrad, wgrad, bgrad = cls.run_linear_postprocess_parallel(
            y_q,
            dgrad,
            wgrad,
            bgrad,
            parallel_mode,
            sequence_parallel,
            tp_size,
            tp_group,
        )

        return y_q, dgrad, wgrad, bgrad

    def compare_recipe(
        self,
        recipe1,
        recipe2,
        batch_size,
        hidden_size,
        out_size,
        use_bias,
        seed,
        dtype,
        y_error=0.0,
        dgrad_error=0.0,
        wgrad_error=0.0,
        bgrad_error=0.0,
    ):
        x, w, bias, gradient = self._prepare_data(batch_size, hidden_size, out_size, use_bias, seed=seed, dtype=dtype)

        # recipe1
        y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(x, w, bias, gradient, recipe1)
        # recipe2
        y_q, dgrad, wgrad, bgrad = self.run_linear(x, w, bias, gradient, recipe2)

        # Compare results
        torch.testing.assert_close(y_q.float(), y_q_ref.float(), rtol=0.0, atol=y_error)
        torch.testing.assert_close(dgrad, dgrad_ref, rtol=0.0, atol=dgrad_error)
        torch.testing.assert_close(wgrad, wgrad_ref, rtol=0.0, atol=wgrad_error)
        if use_bias:
            torch.testing.assert_close(bgrad, bgrad_ref, rtol=0.0, atol=bgrad_error)

    def compare_weight_cache(
        self,
        recipe1,
        batch_size,
        hidden_size,
        out_size,
        use_bias,
        seed,
        dtype,
        run_num_steps=10,
        y_error=0.0,
        dgrad_error=0.0,
        wgrad_error=0.0,
        bgrad_error=0.0,
    ):
        x, w, bias, gradient = self._prepare_data(batch_size, hidden_size, out_size, use_bias, seed=seed, dtype=dtype)

        # without weight cache
        y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(
            x,
            w,
            bias,
            gradient,
            recipe1,
            run_num_steps=run_num_steps,
            enable_weight_cache=False,
        )
        # with weight cache
        y_q, dgrad, wgrad, bgrad = self.run_linear(
            x,
            w,
            bias,
            gradient,
            recipe1,
            run_num_steps=run_num_steps,
            enable_weight_cache=True,
        )

        torch.testing.assert_close(y_q, y_q_ref, rtol=0.0, atol=y_error)
        torch.testing.assert_close(dgrad, dgrad_ref, rtol=0.0, atol=dgrad_error)
        torch.testing.assert_close(wgrad, wgrad_ref, rtol=0.0, atol=wgrad_error)
        if use_bias:
            torch.testing.assert_close(bgrad, bgrad_ref, rtol=0.0, atol=bgrad_error)

    def compare_fuse_or_not_fuse_wgrad_accumulation(
        self,
        recipe,
        batch_size,
        hidden_size,
        out_size,
        use_bias,
        seed,
        dtype,
    ):
        x, w, bias, gradient = self._prepare_data(batch_size, hidden_size, out_size, use_bias, seed=seed, dtype=dtype)

        # ref: without fused weight gradient accumulation
        y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = self.run_linear(
            x,
            w,
            bias,
            gradient,
            recipe,
            run_num_steps=100,
            fuse_wgrad_accumulation=False,
        )
        wgrad_ref = torch.cumsum(wgrad_ref, dim=0)

        # with fused weight gradient accumulation
        y_q, dgrad, wgrad, bgrad = self.run_linear(
            x,
            w,
            bias,
            gradient,
            recipe,
            run_num_steps=100,
            fuse_wgrad_accumulation=True,
        )

        # Compare results
        torch.testing.assert_close(y_q.float(), y_q_ref.float(), rtol=0.0, atol=0.0)
        torch.testing.assert_close(dgrad, dgrad_ref, rtol=0.0, atol=0.0)
        torch.testing.assert_close(wgrad, wgrad_ref, rtol=0.0, atol=0.0)
        if use_bias:
            torch.testing.assert_close(bgrad, bgrad_ref, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not HAVE_KITCHEN or not torch.cuda.is_available(),
    reason="Must have kitchen and cuda",
)
class TestGroupedLinear(TestLinearBase):
    def _run_grouped_linear(self, module, num_gemms, seq_len, batch_size, hidden_size, dtype):
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        x = torch.randn(
            (seq_len, batch_size, hidden_size),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        x.retain_grad()

        if num_gemms > 1:
            # split seq_len randomly, so that each split is a multiple of 16
            # and at least one split is 0.
            split_size = 16
            m = seq_len // split_size
            dist = torch.sort(torch.randint(0, m, (num_gemms - 2,))).values.tolist()
            dist.append(dist[-1])  # Manually add a zero
            m_splits = torch.tensor(dist + [m]) - torch.tensor([0] + dist)
            m_splits = m_splits * split_size
            assert m_splits.sum() == seq_len and len(m_splits) == num_gemms
        else:
            m_splits = torch.tensor([seq_len])

        if isinstance(module, GroupedLinear):
            # in real cases, m_splits is split along the tokens, so we multiply batch_size here
            m_splits = m_splits * batch_size
            out = module(x, m_splits.tolist())
        elif isinstance(module, torch.nn.ModuleList):
            out = torch.cat([module[i](inp) for i, inp in enumerate(torch.split(x, m_splits.tolist()))])
        else:
            # in real cases, m_splits is split along the tokens, so we multiply batch_size here
            m_splits = m_splits * batch_size
            out = module(x, m_splits.tolist())

        loss = out.sum()
        loss.backward()

        torch.cuda.synchronize()
        outputs = [out, x.grad]
        for p in module.parameters():
            if p.requires_grad:
                if getattr(p, "main_grad", None) is not None:
                    outputs.append(p.main_grad)
                    assert p.grad is None
                else:
                    outputs.append(p.grad)
        return outputs

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    @pytest.mark.parametrize(
        "hidden_size, out_size",
        [
            (256, 128),
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("num_gemms", [3, 6])
    @pytest.mark.parametrize(
        "recipe",
        [
            Recipe.fp8_per_sub_channel_cutlass,
            Recipe.fp8_per_sub_channel_cublas,
        ],
    )
    @pytest.mark.parametrize("fuse_wgrad_accumulation", [True, False])
    def test_grouped_linear_recipes(
        self,
        recipe,
        num_gemms,
        batch_size,
        hidden_size,
        out_size,
        dtype,
        fuse_wgrad_accumulation,
        use_bias=True,
    ):
        # TODO(xiny): Enable this once cuBLAS fixes the multi-stream bug
        if recipe == Recipe.fp8_per_sub_channel_cublas:
            pytest.skip("cuBLAS multi-stream bug")
        use_bias = use_bias and dtype != torch.float32
        if recipe == Recipe.fp8_per_sub_channel_cutlass:
            use_bias = False
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        seq_len = 2048

        grouped_linear = GroupedLinear(
            num_gemms,
            hidden_size,
            out_size,
            bias=use_bias,
            params_dtype=dtype,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            qlinear_params=recipe(),
        )
        sequential_linear = torch.nn.ModuleList(
            [
                Linear(
                    hidden_size,
                    out_size,
                    bias=use_bias,
                    params_dtype=dtype,
                    device="cuda",
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    qlinear_params=recipe(),
                )
                for _ in range(num_gemms)
            ]
        )

        # Share params
        with torch.no_grad():
            for i in range(num_gemms):
                sequential_linear[i].weight.copy_(getattr(grouped_linear, f"weight{i}"))
                if use_bias:
                    sequential_linear[i].bias.copy_(getattr(grouped_linear, f"bias{i}"))
                if fuse_wgrad_accumulation:
                    weight_i = getattr(grouped_linear, f"weight{i}")
                    weight_i.main_grad = torch.rand_like(weight_i, dtype=torch.float32)
                    sequential_linear[i].weight.main_grad = weight_i.main_grad.clone()

        outputs = self._run_grouped_linear(grouped_linear, num_gemms, seq_len, batch_size, hidden_size, dtype)
        outputs_ref = self._run_grouped_linear(sequential_linear, num_gemms, seq_len, batch_size, hidden_size, dtype)

        # Shoule be bit-wise match
        for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
            torch.testing.assert_close(o, o_ref, rtol=0, atol=0)

    @pytest.mark.skipif(
        not HAVE_TE,
        reason="Must have transformer engine",
    )
    def test_grouped_linear_with_te(self):
        recipe = Recipe.fp8_per_sub_channel_cutlass
        num_gemms = 3
        batch_size = 1
        hidden_size = 64
        out_size = 336
        dtype = torch.bfloat16
        fuse_wgrad_accumulation = True
        use_bias = True

        # TODO(xiny): Enable this once cuBLAS fixes the multi-stream bug
        if recipe == Recipe.fp8_per_sub_channel_cublas:
            pytest.skip("cuBLAS multi-stream bug")
        use_bias = use_bias and dtype != torch.float32
        if recipe == Recipe.fp8_per_sub_channel_cutlass:
            use_bias = False
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        seq_len = 2048

        grouped_linear = GroupedLinear(
            num_gemms,
            hidden_size,
            out_size,
            bias=use_bias,
            params_dtype=dtype,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            qlinear_params=recipe(),
        )
        te_grouped_linear = te.pytorch.GroupedLinear(
            num_gemms,
            hidden_size,
            out_size,
            bias=use_bias,
            params_dtype=dtype,
            device="cuda",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )

        # Share params
        with torch.no_grad():
            for i in range(num_gemms):
                te_weight = getattr(te_grouped_linear, f"weight{i}")
                te_weight.copy_(getattr(grouped_linear, f"weight{i}"))

                if use_bias:
                    te_bias = getattr(te_grouped_linear, f"bias{i}")
                    te_bias.copy_(getattr(grouped_linear, f"bias{i}"))

                if fuse_wgrad_accumulation:
                    weight_i = getattr(grouped_linear, f"weight{i}")
                    weight_i.main_grad = torch.rand_like(weight_i, dtype=torch.float32)
                    te_weight.main_grad = weight_i.main_grad.clone()

                torch.testing.assert_close(te_weight, getattr(grouped_linear, f"weight{i}"), rtol=0, atol=0)

                if use_bias:
                    torch.testing.assert_close(te_bias, getattr(grouped_linear, f"bias{i}"), rtol=0, atol=0)

                if fuse_wgrad_accumulation:
                    torch.testing.assert_close(te_weight.main_grad, weight_i.main_grad, rtol=0, atol=0)

        outputs = self._run_grouped_linear(grouped_linear, num_gemms, seq_len, batch_size, hidden_size, dtype)
        outputs_ref = self._run_grouped_linear(te_grouped_linear, num_gemms, seq_len, batch_size, hidden_size, dtype)

        # Shoule be bit-wise match
        for i, (o, o_ref) in enumerate(zip(outputs, outputs_ref)):
            torch.testing.assert_close(o, o_ref, rtol=0.7, atol=0.7)
