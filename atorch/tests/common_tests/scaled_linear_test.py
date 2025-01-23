import unittest

import pytest
import torch
import torch.nn.functional as F

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.modules.fp8 import ScaledLinear, get_fp8_module_count, set_linear_modules_precision
from atorch.modules.fp8.quantize import (
    get_fp8_quantize_underflows,
    get_linear_axiswise_quantize_params,
    get_linear_tileblock_quantize_params,
)
from atorch.tests.toy_modules.toy_module import ToyDataset, ToyModel, loss_func, optim_func, prepare_input, run_train

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
    not torch.cuda.is_available() or not is_fp8_available(),
    "only on gpu with fp8 supported",
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
            dataloader_args={"batch_size": batch_size},
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

    @unittest.skipIf(not atorch.modules.fp8.quantize.fbgemm_available, "fbgemm_gpu required")
    def test_scaled_linear_with_auto_accelerate_with_fbgemm(self):
        scale_method = "axiswise"
        quantization_method = "fbgemm"
        compute_method = "pytorch"
        self._run_auto_accelerate_test(scale_method, quantization_method, compute_method)

    def test_forward(self):
        M, K, N = (512, 1024, 1024)
        dtype = torch.bfloat16
        input = torch.randn([M, K], dtype=dtype, device="cuda")
        input[0][0] *= 100
        input[1][1] = 100
        model = ScaledLinear(K, N, bias=False, device="cuda", dtype=dtype, quantize_params="tensorwise")
        qparams = get_linear_tileblock_quantize_params(128)
        model2 = ScaledLinear(K, N, bias=False, device="cuda", dtype=dtype, quantize_params=qparams)
        model.set_use_fp8(False)
        with torch.no_grad():
            bf16_res = model(input)
            model.set_use_fp8(True)
            model2.weight.data.copy_(model.weight)  # sync weight
            # compare tensorwise, tileblockwise results
            tensorwise_res = model(input)
            tileblock_res = model2(input)
            tensorwise_mse = torch.nn.functional.mse_loss(bf16_res, tensorwise_res)
            tileblock_mse = torch.nn.functional.mse_loss(bf16_res, tileblock_res)
            self.assertTrue(tensorwise_mse >= tileblock_mse)
