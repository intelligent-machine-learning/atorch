from unittest.mock import MagicMock, patch

import pytest
import torch

from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import utils as megatron_utils
    from megatron.legacy import model as megatron_legacy_model

    from atorch.trainer.utils import training_log
else:
    megatron_utils = MagicMock()
    megatron_legacy_model = MagicMock()


MOCK_BASE = "atorch.trainer.utils"


def _setup_training_log_mocks(
    mock_datetime,
    mock_mpu,
    mock_report_theoretical_memory,
    mock_report_memory,
    mock_num_fp_ops,
    mock_mem_stats,
    mock_version_check,
    mock_print_rank_last,
    mock_get_num_microbatches,
    mock_get_one_logger,
    mock_get_wandb_writer,
    mock_get_tensorboard_writer,
    mock_get_timers,
    mock_get_args,
):
    """Helper to setup common mocks for training_log tests."""
    # Setup mocks
    mock_args = MagicMock()
    mock_get_args.return_value = mock_args

    mock_timers_instance = MagicMock()
    mock_timers_instance.log = MagicMock()
    interval_timer = MagicMock()
    interval_timer.elapsed.return_value = 1.0
    mock_timers_instance.return_value = interval_timer
    mock_get_timers.return_value = mock_timers_instance

    mock_writer = MagicMock()
    mock_get_tensorboard_writer.return_value = mock_writer

    mock_datetime.now.return_value.strftime.return_value = "2023-01-01 00:00:00"

    # Common args
    mock_args.log_interval = 10
    mock_args.use_local_sgd = False
    mock_args.num_experts = None
    mock_args.mtp_num_layers = None
    mock_args.log_throughput = False
    mock_args.decoupled_lr = None
    mock_args.data_parallel_size = 1
    mock_args.micro_batch_size = 1
    mock_args.world_size = 1
    mock_args.consumed_train_samples = 100
    mock_args.train_iters = 100
    mock_args.tensorboard_log_interval = 5
    mock_args.skipped_train_samples = 0
    mock_args.log_memory_to_tensorboard = False

    mock_mpu.is_pipeline_first_stage.return_value = True
    mock_mpu.is_pipeline_last_stage.return_value = True

    # Dummy inputs for training_log
    loss_dict = {"lm_loss": torch.tensor(1.0)}
    total_loss_dict = {}
    common_kwargs = dict(
        loss_dict=loss_dict,
        total_loss_dict=total_loss_dict,
        learning_rate=0.001,
        decoupled_learning_rate=None,
        loss_scale=1.0,
        report_memory_flag=False,
        skipped_iter=0,
        grad_norm=None,
        params_norm=None,
        num_zeros_in_grad=None,
        params_std=None,
        custom_metrics=None,
    )

    # To enter the main logging block, `iteration % args.log_interval` must be 0.
    # We set `log_interval` to 10.
    iteration = 10

    return mock_args, mock_timers_instance, common_kwargs, iteration


@pytest.mark.skipif(not is_megatron_lm_available(), reason="Megatron-LM not available.")
@patch(f"{MOCK_BASE}.get_args")
@patch(f"{MOCK_BASE}.get_timers")
@patch(f"{MOCK_BASE}.get_tensorboard_writer")
@patch(f"{MOCK_BASE}.get_wandb_writer")
@patch(f"{MOCK_BASE}.get_one_logger")
@patch(f"{MOCK_BASE}.get_num_microbatches", return_value=1)
@patch(f"{MOCK_BASE}.print_rank_last")
@patch(f"{MOCK_BASE}.is_megatron_version_bigger_than", return_value=False)
@patch(f"{MOCK_BASE}.torch.cuda.memory_stats", return_value={})
@patch(f"{MOCK_BASE}.num_floating_point_operations", return_value=1e12)
@patch(f"{MOCK_BASE}.report_memory")
@patch(f"{MOCK_BASE}.report_theoretical_memory")
@patch(f"{MOCK_BASE}.mpu")
@patch(f"{MOCK_BASE}.datetime")
def test_timers_log_when_tensorboard_is_disabled(
    mock_datetime,
    mock_mpu,
    mock_report_theoretical_memory,
    mock_report_memory,
    mock_num_fp_ops,
    mock_mem_stats,
    mock_version_check,
    mock_print_rank_last,
    mock_get_num_microbatches,
    mock_get_one_logger,
    mock_get_wandb_writer,
    mock_get_tensorboard_writer,
    mock_get_timers,
    mock_get_args,
):
    """Case 1: Test timers.log() is called when log_timers_to_tensorboard is False."""
    mock_args, mock_timers_instance, common_kwargs, iteration = _setup_training_log_mocks(
        mock_datetime,
        mock_mpu,
        mock_report_theoretical_memory,
        mock_report_memory,
        mock_num_fp_ops,
        mock_mem_stats,
        mock_version_check,
        mock_print_rank_last,
        mock_get_num_microbatches,
        mock_get_one_logger,
        mock_get_wandb_writer,
        mock_get_tensorboard_writer,
        mock_get_timers,
        mock_get_args,
    )

    mock_args.log_timers_to_tensorboard = False
    training_log(iteration=iteration, **common_kwargs)
    mock_timers_instance.log.assert_called_once()


@pytest.mark.skipif(not is_megatron_lm_available(), reason="Megatron-LM not available.")
@patch(f"{MOCK_BASE}.get_args")
@patch(f"{MOCK_BASE}.get_timers")
@patch(f"{MOCK_BASE}.get_tensorboard_writer")
@patch(f"{MOCK_BASE}.get_wandb_writer")
@patch(f"{MOCK_BASE}.get_one_logger")
@patch(f"{MOCK_BASE}.get_num_microbatches", return_value=1)
@patch(f"{MOCK_BASE}.print_rank_last")
@patch(f"{MOCK_BASE}.is_megatron_version_bigger_than", return_value=False)
@patch(f"{MOCK_BASE}.torch.cuda.memory_stats", return_value={})
@patch(f"{MOCK_BASE}.num_floating_point_operations", return_value=1e12)
@patch(f"{MOCK_BASE}.report_memory")
@patch(f"{MOCK_BASE}.report_theoretical_memory")
@patch(f"{MOCK_BASE}.mpu")
@patch(f"{MOCK_BASE}.datetime")
def test_timers_log_when_not_tensorboard_iteration(
    mock_datetime,
    mock_mpu,
    mock_report_theoretical_memory,
    mock_report_memory,
    mock_num_fp_ops,
    mock_mem_stats,
    mock_version_check,
    mock_print_rank_last,
    mock_get_num_microbatches,
    mock_get_one_logger,
    mock_get_wandb_writer,
    mock_get_tensorboard_writer,
    mock_get_timers,
    mock_get_args,
):
    """Case 2: Test timers.log() is called when it's not a tensorboard log iteration."""
    mock_args, mock_timers_instance, common_kwargs, iteration = _setup_training_log_mocks(
        mock_datetime,
        mock_mpu,
        mock_report_theoretical_memory,
        mock_report_memory,
        mock_num_fp_ops,
        mock_mem_stats,
        mock_version_check,
        mock_print_rank_last,
        mock_get_num_microbatches,
        mock_get_one_logger,
        mock_get_wandb_writer,
        mock_get_tensorboard_writer,
        mock_get_timers,
        mock_get_args,
    )

    mock_args.log_timers_to_tensorboard = True
    mock_args.tensorboard_log_interval = 3  # 10 % 3 != 0
    training_log(iteration=iteration, **common_kwargs)
    mock_timers_instance.log.assert_called_once()


@pytest.mark.skipif(not is_megatron_lm_available(), reason="Megatron-LM not available.")
@patch(f"{MOCK_BASE}.get_args")
@patch(f"{MOCK_BASE}.get_timers")
@patch(f"{MOCK_BASE}.get_tensorboard_writer")
@patch(f"{MOCK_BASE}.get_wandb_writer")
@patch(f"{MOCK_BASE}.get_one_logger")
@patch(f"{MOCK_BASE}.get_num_microbatches", return_value=1)
@patch(f"{MOCK_BASE}.print_rank_last")
@patch(f"{MOCK_BASE}.is_megatron_version_bigger_than", return_value=False)
@patch(f"{MOCK_BASE}.torch.cuda.memory_stats", return_value={})
@patch(f"{MOCK_BASE}.num_floating_point_operations", return_value=1e12)
@patch(f"{MOCK_BASE}.report_memory")
@patch(f"{MOCK_BASE}.report_theoretical_memory")
@patch(f"{MOCK_BASE}.mpu")
@patch(f"{MOCK_BASE}.datetime")
def test_timers_no_log_on_tensorboard_iteration(
    mock_datetime,
    mock_mpu,
    mock_report_theoretical_memory,
    mock_report_memory,
    mock_num_fp_ops,
    mock_mem_stats,
    mock_version_check,
    mock_print_rank_last,
    mock_get_num_microbatches,
    mock_get_one_logger,
    mock_get_wandb_writer,
    mock_get_tensorboard_writer,
    mock_get_timers,
    mock_get_args,
):
    """Case 3: Test timers.log() is NOT called on a tensorboard log iteration."""
    mock_args, mock_timers_instance, common_kwargs, iteration = _setup_training_log_mocks(
        mock_datetime,
        mock_mpu,
        mock_report_theoretical_memory,
        mock_report_memory,
        mock_num_fp_ops,
        mock_mem_stats,
        mock_version_check,
        mock_print_rank_last,
        mock_get_num_microbatches,
        mock_get_one_logger,
        mock_get_wandb_writer,
        mock_get_tensorboard_writer,
        mock_get_timers,
        mock_get_args,
    )

    mock_args.log_timers_to_tensorboard = True
    mock_args.tensorboard_log_interval = 5  # 10 % 5 == 0
    training_log(iteration=iteration, **common_kwargs)
    mock_timers_instance.log.assert_not_called()


@pytest.mark.skipif(not is_megatron_lm_available(), reason="Megatron-LM not available.")
@patch(f"{MOCK_BASE}.get_args")
@patch(f"{MOCK_BASE}.get_timers")
@patch(f"{MOCK_BASE}.get_tensorboard_writer")
@patch(f"{MOCK_BASE}.get_wandb_writer")
@patch(f"{MOCK_BASE}.get_one_logger")
@patch(f"{MOCK_BASE}.get_num_microbatches", return_value=1)
@patch(f"{MOCK_BASE}.print_rank_last")
@patch(f"{MOCK_BASE}.is_megatron_version_bigger_than", return_value=False)
@patch(f"{MOCK_BASE}.torch.cuda.memory_stats", return_value={})
@patch(f"{MOCK_BASE}.num_floating_point_operations", return_value=1e12)
@patch(f"{MOCK_BASE}.report_memory")
@patch(f"{MOCK_BASE}.report_theoretical_memory")
@patch(f"{MOCK_BASE}.mpu")
@patch(f"{MOCK_BASE}.datetime")
@patch("atorch.local_sgd.megatron.parallel_state.get_non_data_parallel_group")
@patch.object(megatron_utils, "is_float8tensor", create=True)
@patch.object(megatron_legacy_model, "Float16Module", create=True)
def test_timers_log_for_local_sgd(
    mock_float16module,
    mock_is_float8tensor,
    mock_get_group,
    mock_datetime,
    mock_mpu,
    mock_report_theoretical_memory,
    mock_report_memory,
    mock_num_fp_ops,
    mock_mem_stats,
    mock_version_check,
    mock_print_rank_last,
    mock_get_num_microbatches,
    mock_get_one_logger,
    mock_get_wandb_writer,
    mock_get_tensorboard_writer,
    mock_get_timers,
    mock_get_args,
):
    """Case 4: Test timers.log() is called correctly for local SGD."""
    mock_args, mock_timers_instance, common_kwargs, iteration = _setup_training_log_mocks(
        mock_datetime,
        mock_mpu,
        mock_report_theoretical_memory,
        mock_report_memory,
        mock_num_fp_ops,
        mock_mem_stats,
        mock_version_check,
        mock_print_rank_last,
        mock_get_num_microbatches,
        mock_get_one_logger,
        mock_get_wandb_writer,
        mock_get_tensorboard_writer,
        mock_get_timers,
        mock_get_args,
    )

    mock_get_group.return_value = "dummy_group"
    mock_args.use_local_sgd = True
    mock_args.log_timers_to_tensorboard = False
    training_log(iteration=iteration, **common_kwargs)
    mock_timers_instance.log.assert_called_once_with(
        [
            "forward-backward",
            "forward-compute",
            "backward-compute",
            "batch-generator",
            "forward-recv",
            "forward-send",
            "backward-recv",
            "backward-send",
            "forward-send-forward-recv",
            "forward-send-backward-recv",
            "backward-send-forward-recv",
            "backward-send-backward-recv",
            "forward-backward-send-forward-backward-recv",
            "layernorm-grads-all-reduce",
            "embedding-grads-all-reduce",
            "all-grads-sync",
            "params-all-gather",
            "optimizer-copy-to-main-grad",
            "optimizer-unscale-and-check-inf",
            "optimizer-clip-main-grad",
            "optimizer-count-zeros",
            "optimizer-inner-step",
            "optimizer-copy-main-to-model-params",
            "optimizer",
        ],
        normalizer=mock_args.log_interval,
        process_group="dummy_group",
    )
