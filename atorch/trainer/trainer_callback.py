import json
from dataclasses import dataclass, fields

from tqdm import tqdm
from transformers.trainer_callback import CallbackHandler, TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import IntervalStrategy

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.atorch_args import AtorchArguments
from atorch.trainer.utils import DistributedType
from atorch.trainer.utils import IntervalStrategy as AtorchIntervalStrategy
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    try:
        from megatron.training import get_current_global_batch_size
    except ImportError:
        from megatron.core.num_microbatches_calculator import get_current_global_batch_size


@dataclass
class AtorchTrainerState(TrainerState):
    steps_in_epoch: int = 0
    current_step_in_epoch: int = 0
    consumed_train_samples: int = 0
    consumed_train_tokens: int = 0

    @classmethod
    def load_from_json(cls, json_path: str, origin_train_state_to_fill=None):
        """Create an instance from the content of `json_path`."""

        # Forward compatibility: keys in AtorchTrainerState may be extended in the future,
        # so it is necessary to verify the keys in trainer_state.json
        with open(json_path, "r", encoding="utf-8") as f:
            state_dict = json.loads(f.read())

        field_names = {field.name for field in fields(cls)}

        if set(state_dict.keys()) != field_names:
            logger.info(
                f"Keys {set(state_dict.keys()) - field_names} in {json_path} "
                "will not be used in this version atorch."
            )
            state_dict = {k: v for k, v in state_dict.items() if k in field_names}

        new_loaded_train_state = cls(**state_dict)

        if origin_train_state_to_fill is not None:
            origin_train_state_to_fill.replace(new_loaded_train_state)
            return origin_train_state_to_fill
        else:
            return new_loaded_train_state

    def replace(self, new_state):
        for field in fields(new_state):
            setattr(self, field.name, getattr(new_state, field.name))


@dataclass
class AtorchTrainerControl(TrainerControl):
    should_test: bool = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_test = False
        super()._new_step()


class AtorchTrainerCallback(TrainerCallback):
    """
    Add extra events for AtorchTrainerCallback.
    """

    def on_evaluate_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        """
        Event called at the beginning of an evaluation phase.
        """
        pass

    def on_save_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        """
        Event called at the beginning of saving.
        """
        pass

    def on_predict_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        """
        Event called at the beginning of prediction.
        """
        pass


class AtorchCallbackHandler(CallbackHandler):
    """
    Add extra events for transformers CallbackHandler.
    """

    def on_evaluate_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        return self.call_event_safely("on_evaluate_begin", args, state, control, **kwargs)

    def on_save_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        return self.call_event_safely("on_save_begin", args, state, control, **kwargs)

    def on_predict_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        return self.call_event_safely("on_predict_begin", args, state, control, **kwargs)

    def on_prediction_step(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_prediction_step", args, state, control, **kwargs)

    def call_event_safely(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            if not hasattr(callback, event):
                continue

            result = getattr(callback, event)(
                args,
                state,
                control,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class FlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: AtorchArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == IntervalStrategy.STEPS
            and state.global_step % args.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and args.save_steps > 0
            and state.global_step % args.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: AtorchArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True
        elif (
            args.save_at_specific_epoch is not None
            and state.epoch is not None
            and round(state.epoch) in args.save_at_specific_epoch
        ):
            control.should_save = True

        return control


class FlowCallbackV2(AtorchTrainerCallback):
    """
    A [`AtorchTrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        control.should_test = False

    def on_step_end(self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == AtorchIntervalStrategy.STEPS and state.global_step % args.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.evaluation_strategy == AtorchIntervalStrategy.STEPS
            and state.global_step % args.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Test
        if (
            args.test_strategy == AtorchIntervalStrategy.STEPS
            and state.global_step % args.test_steps == 0
            and args.test_delay <= state.global_step
        ):
            control.should_test = True

        if args.distributed_state.distributed_type == DistributedType.MEGATRON:
            global_batch_size = get_current_global_batch_size()
        else:
            global_batch_size = args.global_train_batch_size

        def _judge_save_ckpt_by_samples():
            should_save = False
            if state.consumed_train_samples % args.save_samples == 0:
                should_save = True
            elif (
                state.consumed_train_samples % args.save_samples <= global_batch_size / 2
                or (args.save_samples - state.consumed_train_samples % args.save_samples) < global_batch_size / 2
            ):
                should_save = True
            return should_save

        # Save
        if (
            (
                args.save_strategy == AtorchIntervalStrategy.STEPS
                and args.save_steps > 0
                and state.global_step % args.save_steps == 0
            )
            or (
                args.save_strategy == AtorchIntervalStrategy.SAMPLES
                and args.save_samples > 0
                and _judge_save_ckpt_by_samples()
            )
            or (
                # extra save frequency in each epoch
                args.extra_save_frequency_in_epoch is not None
                and state.current_step_in_epoch in args.extra_save_frequency_in_epoch
            )
        ):
            control.should_save = True
            # Extra judge about test.
            if args.test_on_save:
                control.should_test = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        # Log
        if args.logging_strategy == AtorchIntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == AtorchIntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Test
        if args.test_strategy == AtorchIntervalStrategy.EPOCH and args.test_delay <= state.epoch:
            control.should_test = True

        # Save
        if args.save_strategy == AtorchIntervalStrategy.EPOCH:
            control.should_save = True

        return control

    def on_evaluate_begin(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs
    ):
        if "eval_type" in kwargs:
            args.eval_type = kwargs["eval_type"]

    def on_evaluate(self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs):
        if args.eval_type == "test":
            control.should_test = False
        if args.eval_type is not None:
            args.eval_type = None


# TODO: Unify the training bar and prediction bar
class PredictCallback(AtorchTrainerCallback):
    """
    A [`AtorchTrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.prediction_bar = None

    def should_print(self, args: AtorchTrainingArgs):
        should_print = False
        if args.distributed_state.distributed_type == DistributedType.MEGATRON:
            should_print = args.is_last_process
        else:
            should_print = args.is_main_process
        return should_print

    def on_prediction_step(
        self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, eval_iters=0, **kwargs
    ):
        if self.should_print(args) and eval_iters > 0:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=eval_iters, dynamic_ncols=True)
            self.prediction_bar.update(1)

    def on_evaluate(self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs):
        if self.should_print(args):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: AtorchTrainerControl, **kwargs):
        if self.should_print(args):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None
