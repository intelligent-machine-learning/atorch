import unittest

import pytest
import torch

from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.tests.toy_modules.toy_module import ToyDataset, ToyModel, loss_func, optim_func, prepare_input, run_train

pytestmark = pytest.mark.fp8


@unittest.skipIf(
    not torch.cuda.is_available() or not is_fp8_available(),
    "only on gpu with fp8 supported",
)
class TestPrecisionSwitchLinear(unittest.TestCase):
    def test_switch_linear(self):
        infeature = 64
        outfeature = 32
        batch = 32 * infeature

        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe

        from atorch.modules.fp8 import PrecisionSwitchableLinear
        from atorch.modules.fp8.precision_switchable_linear import _patch_te_linear_backward

        fp8_recipe = recipe.DelayedScaling()
        layer = PrecisionSwitchableLinear(infeature, outfeature, device="cuda")
        data = torch.rand([batch, infeature], device="cuda")

        res = layer(data)
        layer.switch_precision("fp8")
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            res = layer(data)
            layer.switch_precision()
            res = layer(data)
            layer.switch_precision("fp8")
            res = layer(data)
            self.assertTrue(res.dtype == torch.float32)
        torch.cuda.synchronize()
        state_dict = layer.state_dict()
        self.assertTrue("_extra_state" in state_dict.keys())
        layer.load_state_dict(state_dict)
        state_dict.pop("_extra_state")
        layer.load_state_dict(state_dict, strict=False)

        # test_pre_cast_input_fp8_current_scaling
        layer.switch_precision("original")
        data.requires_grad = True
        data.grad = None
        layer.weight.grad = None
        layer.bias.grad = None
        res = layer(data)
        loss = res.sum() / 100
        loss.backward()
        fp32_data_grad = data.grad.detach()
        fp32_weight_grad = layer.weight.grad.detach()

        layer.switch_precision("fp8")
        data.grad = None
        layer.weight.grad = None
        layer.bias.grad = None
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            res = layer(data)
            loss = res.sum() / 100
            loss.backward()
        auto_fp8_data_grad = data.grad.detach()
        auto_fp8_weight_grad = layer.weight.grad.detach()

        layer.set_pre_cast_input_fp8_current_scaling(True)
        data.grad = None
        layer.weight.grad = None
        layer.bias.grad = None
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            res = layer(data)
            loss = res.sum() / 100
            loss.backward()
        precast_fp8_data_grad = data.grad.detach()
        precast_fp8_weight_grad = layer.weight.grad.detach()

        auto_data_mse = torch.nn.functional.mse_loss(fp32_data_grad, auto_fp8_data_grad).to("cpu").item()
        auto_weight_mse = torch.nn.functional.mse_loss(fp32_weight_grad, auto_fp8_weight_grad).to("cpu").item()
        precast_data_mse = torch.nn.functional.mse_loss(fp32_data_grad, precast_fp8_data_grad).to("cpu").item()
        precast_weight_mse = torch.nn.functional.mse_loss(fp32_weight_grad, precast_fp8_weight_grad).to("cpu").item()
        _patch_te_linear_backward(enable=False)
        self.assertTrue(auto_data_mse >= precast_data_mse)
        self.assertTrue(auto_weight_mse >= precast_weight_mse)

    def test_switch_linear_with_auto_accelerate(self):
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe

        from atorch.modules.fp8 import (
            get_fp8_module_count,
            set_linear_modules_pre_cast_input_fp8_current_scaling,
            set_linear_modules_precision,
        )
        from atorch.modules.fp8.precision_switchable_linear import _patch_te_linear_backward

        in_feature = 32
        out_feature = 32
        layer_num = 8
        batch_size = 16
        model = ToyModel(in_features=in_feature, out_features=out_feature, layer_num=layer_num)
        data_num = batch_size * 4
        dataset = ToyDataset(data_num, data_size=(in_feature,), output_size=(out_feature,))

        strategy = [("fp8", {"precision_switchable": True, "include": ("linears,")})]

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

        m_model, m_optim, m_dataloader, m_loss_func, m_prepare_input = (
            result.model,
            result.optim,
            result.dataloader,
            result.loss_func,
            result.prepare_input,
        )

        m_model.linears[1].set_pre_cast_input_fp8_current_scaling(True)
        # linear.2 not use fp8
        m_model.linears[2].switch_precision("original")

        fp8_recipe = recipe.DelayedScaling()

        device = "cuda"
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device)

        m_count = get_fp8_module_count(m_model)
        self.assertTrue(m_count, (layer_num - 1, 0, 0, layer_num, layer_num - 1))

        fp8_include_name_pattern = "{layers.1|layer.3}"
        set_linear_modules_precision(
            m_model,
            fp8_include_name_pattern=fp8_include_name_pattern,
            precision="original",
            verbose=True,
        )
        # now linear.1, linear.2, linear.3 not use fp8
        m_count = get_fp8_module_count(m_model)
        self.assertTrue(m_count, (layer_num - 3, 0, 0, layer_num, layer_num - 3))

        fp8_include_name_pattern = "{layers}"
        fp8_exclude_name_pattern = "{layers.1|layer.3}"
        set_linear_modules_precision(
            m_model,
            fp8_include_name_pattern=fp8_include_name_pattern,
            fp8_exclude_name_pattern=fp8_exclude_name_pattern,
            precision="fp8",
            verbose=True,
        )
        # now linear.1, linear.3 not use fp8
        set_linear_modules_pre_cast_input_fp8_current_scaling(m_model)
        m_count = get_fp8_module_count(m_model)
        self.assertTrue(m_count, (layer_num - 2, 0, 0, layer_num, layer_num - 2))

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device)
        _patch_te_linear_backward(enable=False)
