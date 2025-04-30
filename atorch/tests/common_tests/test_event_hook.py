import sys
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)

from transformers.trainer_callback import TrainerControl  # noqa: E402

from atorch.trainer.args import AtorchTrainingArgs  # noqa: E402
from atorch.trainer.trainer_callback import AtorchCallbackHandler, AtorchTrainerState  # noqa: E402
from atorch.utils.version import torch_version  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
@pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")  # type: ignore
def test_event_hook():
    # ensure the module is not imported
    if "dlrover.python.training_event" in sys.modules:
        del sys.modules["dlrover.python.training_event"]

    mock_module = MagicMock()
    mock_module.TrainerProcess = MagicMock()

    with patch.dict(
        "sys.modules",
        {"dlrover.python.training_event": mock_module},
    ):
        from atorch.trainer.event_util import get_event_callback

        callback = get_event_callback("test")
        assert callback is not None

        callback_hander = AtorchCallbackHandler([callback], None, None, None, None)

        training_args = MagicMock(spec=AtorchTrainingArgs)
        state = AtorchTrainerState()

        control = TrainerControl()
        callback_hander.on_init_end(training_args, state, control)
        callback_hander.on_train_begin(training_args, state, control)
        callback_hander.on_train_end(training_args, state, control)
        callback_hander.on_epoch_begin(training_args, state, control)
        callback_hander.on_epoch_end(training_args, state, control)
        callback_hander.on_substep_end(training_args, state, control)
        callback_hander.on_step_begin(training_args, state, control)
        callback_hander.on_step_end(training_args, state, control)
        callback_hander.on_evaluate_begin(training_args, state, control)
        callback_hander.on_evaluate(training_args, state, control, {})
        callback_hander.on_predict_begin(training_args, state, control)
        callback_hander.on_predict(training_args, state, control, {})
        callback_hander.on_prediction_step(training_args, state, control)
        callback_hander.on_save_begin(training_args, state, control)
        callback_hander.on_log(training_args, state, control, {})
        callback_hander.on_save(training_args, state, control)

        mock_trainer_process = mock_module.TrainerProcess.return_value
        assert len(mock_trainer_process.method_calls) > 0, "TrainerProcess methods should be called"
