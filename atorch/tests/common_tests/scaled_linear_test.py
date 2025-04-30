import unittest

import pytest
import torch
import torch.nn.functional as F

from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.modules.fp8 import ScaledLinear, get_fp8_module_count, set_linear_modules_precision
from atorch.modules.fp8.quantize import (
    get_fp8_quantize_underflows,
    get_linear_axiswise_quantize_params,
    get_linear_tileblock_quantize_params,
    get_quantize_params,
)
from atorch.tests.toy_modules.toy_module import ToyDataset, ToyModel, loss_func, optim_func, prepare_input, run_train
from atorch.utils.import_util import is_deep_gemm_available
from atorch.utils.version import torch_version

pytestmark = pytest.mark.fp8


def run_scaled_linear(infeatures, outfeatures, batchsize, bias=True, dtype=torch.bfloat16, block_size=128):
    s_linear_tensorwise = ScaledLinear(infeatures, outfeatures, bias=bias, device="cuda", dtype=dtype)
    s_linear_axiswise = ScaledLinear(
        infeatures,
        outfeatures,
        bias=bias,
        device="cuda",
        dtype=dtype,
        quantize_params=get_linear_axiswise_quantize_params(),
    )
    s_linear_tileblock = ScaledLinear(
        infeatures,
        outfeatures,
        bias=bias,
        device="cuda",
        dtype=dtype,
        quantize_params=get_linear_tileblock_quantize_params(block_size),
    )
    o_linear = torch.nn.Linear(infeatures, outfeatures, bias=bias, device="cuda", dtype=dtype)

    models = [o_linear, s_linear_tensorwise, s_linear_axiswise, s_linear_tileblock]
    names = ["bf16", "tensorwise", "axiswise", "tileblock"]
    num = len(models)

    inputs = [torch.randn([batchsize, infeatures], device="cuda", dtype=dtype, requires_grad=True) for _ in range(num)]

    with torch.no_grad():
        for i in range(1, num):
            models[i].weight.data.copy_(models[0].weight)
            inputs[i].data.copy_(inputs[0])
            if bias:
                models[i].bias.data.copy_(models[0].bias)

    def _train(layer, input):
        res = layer(input)
        loss = torch.sum(res) / batchsize
        loss.backward()
        return loss.detach(), res.detach()

    loss_mses = {}
    res_mses = {}
    w_grad_mses = {}
    x_grad_mses = {}
    bf16_loss = None
    bf16_res = None
    for i in range(num):
        loss, res = _train(models[i], inputs[i])
        if i == 0:
            bf16_loss = loss
            bf16_res = res
        if i > 0:
            with torch.no_grad():
                loss_mse = F.mse_loss(loss, bf16_loss)
                res_mse = F.mse_loss(res, bf16_res)
                w_grad_mse = F.mse_loss(models[i].weight.grad, models[0].weight.grad)
                x_grad_mse = F.mse_loss(inputs[i].grad, inputs[0].grad)
                loss_mses[names[i]] = loss_mse
                res_mses[names[i]] = res_mse
                w_grad_mses[names[i]] = w_grad_mse
                x_grad_mses[names[i]] = x_grad_mse

    return loss_mses, res_mses, w_grad_mses, x_grad_mses


@unittest.skipIf(
    not torch.cuda.is_available() or not is_fp8_available() or torch_version() < (2, 5, 0),  # type: ignore
    "only on gpu with fp8 supported and torch >=2.5",
)
class TestScaledLinear(unittest.TestCase):
    def _check_result(self, mse_values):
        threshold = 0.05
        for item in mse_values:
            for v in item.values():
                self.assertTrue(v < threshold)

    def test_scaled_linear(self):
        infeature = 128 * 2
        outfeature = 128
        batch = 128

        results = run_scaled_linear(infeature, outfeature, batch, bias=True)
        self._check_result(results)
        results = run_scaled_linear(infeature, outfeature, batch, bias=False)
        self._check_result(results)

        # input shape not 16-aligned
        results = run_scaled_linear(infeature, outfeature, 33, bias=True)
        self._check_result(results)

        # outfeature shape not 16-aligned
        results = run_scaled_linear(infeature, 129, batch, bias=True)
        self._check_result(results)

    def test_quantize(self):
        data = torch.rand(128 * 4, 128 * 4, device="cuda", dtype=torch.bfloat16)
        data[0][0] = 10000.0
        results = get_fp8_quantize_underflows(data)
        self.assertTrue(results["tensorwise"] >= results["rowwise"])
        self.assertTrue(results["tensorwise"] >= results["colwise"])
        if "tilewise" in results:
            self.assertTrue(results["tensorwise"] >= results["tilewise"])
            self.assertTrue(results["rowwise"] >= results["tilewise"])
        if "blockwise" in results:
            self.assertTrue(results["tensorwise"] >= results["blockwise"])
        data = torch.rand(128 * 4 - 3, 128 * 4 + 2, device="cuda", dtype=torch.bfloat16)
        data[0][0] = 10000.0
        results = get_fp8_quantize_underflows(data)
        self.assertTrue("tilewise" in results and "blockwise" in results)

    def switch_precision(self):
        layer_num = 6
        result = self._get_auto_accelerate_result(layer_num=layer_num)
        model = result.model
        counts = get_fp8_module_count(model)
        self.assertEqual(counts, (layer_num, layer_num, layer_num, 0, 0))
        fp8_include_name_pattern = "{layers.1|layer.3}"
        set_linear_modules_precision(model, fp8_include_name_pattern=fp8_include_name_pattern, precision="original")
        counts = get_fp8_module_count(model)
        self.assertEqual(counts, (layer_num - 2, layer_num, layer_num - 2, 0, 0))
        fp8_include_name_pattern = "{layers.1|layer.3}"
        fp8_exclude_name_pattern = "{layers.1}"
        set_linear_modules_precision(
            model,
            fp8_include_name_pattern=fp8_include_name_pattern,
            fp8_exclude_name_pattern=fp8_exclude_name_pattern,
            precision="fp8",
        )
        counts = get_fp8_module_count(model)
        self.assertEqual(counts, (layer_num - 1, layer_num, layer_num - 1, 0, 0))

    def _get_auto_accelerate_result(
        self, scale_method="axiswise", quantization_method="pytorch", compute_method="pytorch", layer_num=8
    ):
        in_feature = 64
        out_feature = 128
        batch_size = 128
        model = ToyModel(in_features=in_feature, out_features=out_feature, layer_num=layer_num)
        data_num = batch_size * 4
        dataset = ToyDataset(data_num, data_size=(in_feature,), output_size=(out_feature,))

        strategy = [
            (
                "fp8",
                {
                    "use_te": False,
                    "scale_method": scale_method,
                    "quantization_method": quantization_method,
                    "compute_method": compute_method,
                    "include": ("linears,"),
                },
            ),
            ("amp_native", {"dtype": torch.bfloat16}),
        ]

        status, result, _ = auto_accelerate(
            model,
            dataset=dataset,
            dataloader_args={"batch_size": batch_size, "num_workers": 0},
            prepare_input=prepare_input,
            optim_func=optim_func,
            load_strategy=strategy,
            loss_func=loss_func,
        )
        assert status
        return result

    def _run_auto_accelerate_test(self, scale_method, quantization_method, compute_method):
        result = self._get_auto_accelerate_result(scale_method, quantization_method, compute_method)

        m_model, m_optim, m_dataloader, m_loss_func, m_prepare_input = (
            result.model,
            result.optim,
            result.dataloader,
            result.loss_func,
            result.prepare_input,
        )

        for name, module in m_model.named_modules():
            if "linears.3" in name and isinstance(module, ScaledLinear):
                # set linear.3 ScaledLinear to not use fp8
                module.quantize_params.use_fp8 = False

        device = "cuda"

        run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device)

    def test_scaled_linear_with_auto_accelerate(self):
        scale_method = "tensorwise"
        quantization_method = "pytorch"
        compute_method = "pytorch"
        self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)

        scale_method = "axiswise"
        quantization_method = "pytorch"
        self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)

        scale_method = "tileblock"
        quantization_method = "triton"
        compute_method = "triton"
        self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)

        from atorch.modules.fp8.cuda_kernel import fp8_cutlass_cublas_ops_available

        if fp8_cutlass_cublas_ops_available():
            scale_method = "tileblock"
            quantization_method = "cutlass"
            compute_method = "cutlass"
            self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)
            scale_method = "tileblock"
            quantization_method = "cublas"
            compute_method = "cublas"
            self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)
            if is_deep_gemm_available():
                scale_method = "tileblock"
                quantization_method = "deep_gemm"
                compute_method = "deep_gemm"
                self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)

    def test_forward(self):
        M, K, N = (512, 1024, 1024)
        dtype = torch.bfloat16
        input = torch.randn([M, K], dtype=dtype, device="cuda")
        input[0][0] *= 100
        input[1][1] = 100
        model = ScaledLinear(K, N, bias=False, device="cuda", dtype=dtype, quantize_params="tensorwise")
        tileblock_triton_qparams = get_quantize_params("tileblock", "triton", "triton", 128)
        tileblock_cutlass_qparams = get_quantize_params("tileblock", "cutlass", "cutlass", 128)
        tileblock_cublas_qparams = get_quantize_params("tileblock", "cublas", "cublas", 128)
        model2 = ScaledLinear(K, N, bias=False, device="cuda", dtype=dtype, quantize_params=tileblock_triton_qparams)
        from atorch.modules.fp8.cuda_kernel import fp8_cutlass_cublas_ops_available

        if fp8_cutlass_cublas_ops_available():
            model3 = ScaledLinear(
                K, N, bias=False, device="cuda", dtype=dtype, quantize_params=tileblock_cutlass_qparams
            )
            model4 = ScaledLinear(
                K, N, bias=False, device="cuda", dtype=dtype, quantize_params=tileblock_cublas_qparams
            )
            if is_deep_gemm_available():
                tileblock_deep_gemm_qparams = get_quantize_params("tileblock", "deep_gemm", "deep_gemm", 128)
                model5 = ScaledLinear(
                    K, N, bias=False, device="cuda", dtype=dtype, quantize_params=tileblock_deep_gemm_qparams
                )
            else:
                model5 = None
        else:
            model3 = None
            model4 = None

        model.set_use_fp8(False)
        with torch.no_grad():
            bf16_res = model(input)
            model.set_use_fp8(True)
            model2.weight.data.copy_(model.weight)  # sync weight
            tensorwise_res = model(input)
            tileblock_triton_res = model2(input)
            tensorwise_mse = torch.nn.functional.mse_loss(bf16_res, tensorwise_res)
            tileblock_triton_mse = torch.nn.functional.mse_loss(bf16_res, tileblock_triton_res)
            self.assertTrue(tensorwise_mse >= tileblock_triton_mse)
            if model3 is not None:
                model3.weight.data.copy_(model.weight)
                tileblock_cutlass_res = model3(input)
                tileblock_cutlass_mse = torch.nn.functional.mse_loss(bf16_res, tileblock_cutlass_res)
                self.assertTrue(tensorwise_mse >= tileblock_cutlass_mse)
            if model4 is not None:
                model4.weight.data.copy_(model.weight)
                tileblock_cublas_res = model4(input)
                tileblock_cublas_mse = torch.nn.functional.mse_loss(bf16_res, tileblock_cublas_res)
                self.assertTrue(tensorwise_mse >= tileblock_cublas_mse)
            if model5 is not None:
                model5.weight.data.copy_(model.weight)
                tileblock_deep_gemm_res = model5(input)
                tileblock_deep_gemm_mse = torch.nn.functional.mse_loss(bf16_res, tileblock_deep_gemm_res)
                self.assertTrue(tensorwise_mse >= tileblock_deep_gemm_mse)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_fp8_available() or torch_version() < (2, 5, 0),  # type: ignore
    "only on gpu with fp8 supported and torch >=2.5",
)
class TestFp8Kernels(unittest.TestCase):
    def test_dequant_kernel(self):
        from atorch.modules.fp8.triton_kernel import block_quant, dequant, tile_quant

        size = 4
        tensor = torch.ones([128 * size, 128 * size], device="cuda")
        tensor[1] = 1000
        tensor[2] = -512

        qx, qs = block_quant(tensor)
        block_dequant_tensor = dequant(qx, qs)

        qx, qs = tile_quant(tensor)
        tile_dequant_tensor = dequant(qx, qs)

        self.assertTrue(torch.all(torch.abs(tensor - block_dequant_tensor) >= torch.abs(tensor - tile_dequant_tensor)))
