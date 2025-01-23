import unittest

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.tests.toy_modules.toy_module import ToyDataset, ToyModel, loss_func, optim_func, prepare_input, run_train
from atorch.utils.inspector import TensorInspector

try:
    import transformer_engine  # noqa: F401

    _te_available = True
except ImportError:
    _te_available = False


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
    inspector.register_hooks(m_model, log_tensor_name_pattern="linears")

    device = "cuda"
    run_train(m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device, inspector=inspector)
    summary_writer.close()


class TestTensorInspector(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
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
