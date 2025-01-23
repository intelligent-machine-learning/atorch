import json
from dataclasses import dataclass, fields

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
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

    @classmethod
    def load_from_json(cls, json_path: str):
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
        return cls(**state_dict)


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


class FlowCallbackV2(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: AtorchTrainingArgs, state: AtorchTrainerState, control: TrainerControl, **kwargs):
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

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True

        return control

    def on_epoch_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == AtorchIntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == AtorchIntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True

        # Save
        if args.save_strategy == AtorchIntervalStrategy.EPOCH:
            control.should_save = True

        return control
