import unittest
from contextlib import nullcontext

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.tests.toy_modules.toy_module import ToyDataset, ToyModel, loss_func, optim_func, prepare_input, run_train
from atorch.tests.toy_modules.toy_module_te import get_input as get_te_input
from atorch.tests.toy_modules.toy_module_te import get_model as get_te_model
from atorch.tests.toy_modules.toy_module_te import loss_func as te_loss_func
from atorch.utils.inspector import TensorInspector

try:
    import transformer_engine  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False


def get_fp8_context(is_init, enabled=True):
    if not enabled:
        return nullcontext()
    fp8_format = transformer_engine.common.recipe.Format.E4M3
    fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(fp8_format=fp8_format)
    if is_init:
        context_args = {"enabled": True, "recipe": fp8_recipe}

        fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)
    else:
        fp8_context = transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=None)
    return fp8_context


def run_te_toy_model_with_inspector(hidden_size=256, num_gemms=4, use_fp8_init=False, use_fp8=False):
    fp8_context_init = get_fp8_context(True, use_fp8_init)
    with fp8_context_init:
        model = get_te_model(hidden_size, num_gemms)
    inspector = TensorInspector(2)
    inspector.register_hooks(model, log_tensor_name_pattern="(gg)", te_fp8_check=use_fp8)
    batch_size = 128 * num_gemms
    for _ in range(5):
        inputs = get_te_input(batch_size, hidden_size)
        fp8_context = get_fp8_context(False, use_fp8)
        with fp8_context:
            outputs = model(inputs)
        loss = te_loss_func(inputs, outputs)
        loss.backward()
        inspector.step()


def run_toy_model_with_inspector(
    in_feature=16, out_feature=16, use_fp8=False, use_te=False, precision_switchable=False, summary_writer_items=None
):
    model = ToyModel(in_features=in_feature, out_features=out_feature)
    batch_size = 16 if use_fp8 else 2
    data_num = batch_size * 4
    dataset = ToyDataset(data_num, data_size=(in_feature,), output_size=(out_feature,))

    if precision_switchable and use_te:
        fp8_strategy = (
            "fp8",
            {
                "precision_switchable": True,
                "include": ("linears",),
            },
        )
    elif use_te:
        fp8_strategy = (
            "fp8",
            {
                "include": ("linears",),
            },
        )
    else:  # ScaledLinear
        fp8_strategy = (
            "fp8",
            {
                "use_te": False,
                "include": ("linears",),
            },
        )
    strategy = [fp8_strategy] if use_fp8 else []
    if use_fp8 and not use_te:
        strategy.append("amp_native")

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

    log_tensor_interval = 2

    summary_writer = SummaryWriter("/tmp/_runs/ut_inspector")
    if use_te:
        inspector = TensorInspector(
            log_tensor_interval, rank=0, summary_writer=summary_writer, summary_writer_items=summary_writer_items
        )
    else:
        inspector = TensorInspector(
            log_tensor_interval,
            save_tensor=True,
            save_tensor_dir="/tmp/_save_tensor",
            plot_tensor=True,
            plot_tensor_dir="/tmp/_plot_tensor",
            rank=0,
            summary_writer=summary_writer,
            summary_writer_items=summary_writer_items,
            log_underflows_for_linear=True,
        )
    inspector.register_hooks(m_model, log_tensor_name_pattern="linears", te_fp8_check=True)

    device = "cuda"
    run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device, inspector=inspector)
    inspector.remove_hooks()
    summary_writer.close()


class TestTensorInspector(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or not is_fp8_available(),
        "No gpu available for cuda tests",
    )
    def test_linear_inspector(self):
        run_toy_model_with_inspector()

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_fp8_available() or not _te_available,
        "No fp8 gpu or te available for te tests",
    )
    @pytest.mark.fp8
    def test_te_inspector(self):
        summary_writer_items = ["underflows"]
        run_toy_model_with_inspector(use_fp8=True, use_te=True, summary_writer_items=summary_writer_items)
        run_toy_model_with_inspector(
            use_fp8=True, use_te=True, precision_switchable=True, summary_writer_items=summary_writer_items
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_fp8_available(),
        "No fp8 gpu or te available for scaled linear tests",
    )
    @pytest.mark.fp8
    def test_scaled_linear_inspector(self):
        summary_writer_items = ["tensorwise_underflows", "rowwise_underflows"]
        run_toy_model_with_inspector(use_fp8=True, use_te=False, summary_writer_items=summary_writer_items)

    @unittest.skipIf(
        not torch.cuda.is_available() or not is_fp8_available(),
        "No fp8 gpu or te available for scaled linear tests",
    )
    @pytest.mark.fp8
    def test_te_module_inspector(self):
        # init fp8, compute fp8
        run_te_toy_model_with_inspector(use_fp8_init=True, use_fp8=True)
        # init not fp8, compute fp8
        run_te_toy_model_with_inspector(use_fp8_init=False, use_fp8=True)
