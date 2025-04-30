import dataclasses

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.trainer_callback import AtorchTrainerCallback, AtorchTrainerState, TrainerControl, TrainerState
from atorch.utils.import_util import is_megatron_lm_available, is_training_event_available


class _TrainerEventCallback(AtorchTrainerCallback):
    """
    Exporter Tariner Event
    """

    def __init__(self, target: str):

        try:
            from dlrover.python.training_event import TrainerProcess  # type: ignore

            self.trainer_process = TrainerProcess(target=target)
        except Exception:
            self.trainer_process = None

        self._train = None
        self._epoch = None
        self._step = None
        self._evaluate = None
        self._predict = None
        self._save = None

    def on_init_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            state_dict = dataclasses.asdict(state)
            args_dict = args.to_dict()

            self.trainer_process.init_end(args=args_dict, state=state_dict)
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_train_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            args_dict = args.to_dict()
            state_dict = dataclasses.asdict(state)
            train_span = self.trainer_process.train(args=args_dict, state=state_dict)

            # export megatron args if available
            try:
                if is_megatron_lm_available():
                    from megatron.training.global_vars import get_args

                    train_span.extra_args(megatron_args=vars(get_args()))
            except Exception:
                pass

            self._train = train_span.begin()

        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_train_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._train is not None:
                state_dict = dataclasses.asdict(state)
                args_dict = args.to_dict()
                self._train.extra_args(state=state_dict, args=args_dict).end()
                self._train = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_epoch_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            state_dict = dataclasses.asdict(state)
            args_dict = args.to_dict()
            self._epoch = self.trainer_process.epoch(state=state_dict, args=args_dict).begin()

        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_epoch_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._epoch is not None:
                state_dict = dataclasses.asdict(state)
                args_dict = args.to_dict()
                self._epoch.extra_args(state=state_dict, args=args_dict).end()
                self._epoch = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_step_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:

            extra_dict = {
                "epoch": state.epoch,
                "total_flos": state.total_flos,
            }
            if isinstance(state, AtorchTrainerState):
                extra_dict.update(
                    {
                        "steps_in_epoch": state.steps_in_epoch,
                        "current_step_in_epoch": state.current_step_in_epoch,
                        "consumed_train_samples": state.consumed_train_samples,
                        "consumed_train_tokens": state.consumed_train_tokens,
                    }
                )

            self._step = self.trainer_process.step(global_step=state.global_step).extra_dict(extra_dict).begin()
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_substep_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            self.trainer_process.substep(global_step=state.global_step, total_flos=state.total_flos)
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_step_end(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._step is not None:
                extra_dict = {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "total_flos": state.total_flos,
                }
                if isinstance(state, AtorchTrainerState):
                    extra_dict.update(
                        {
                            "steps_in_epoch": state.steps_in_epoch,
                            "current_step_in_epoch": state.current_step_in_epoch,
                            "consumed_train_samples": state.consumed_train_samples,
                            "consumed_train_tokens": state.consumed_train_tokens,
                        }
                    )
                self._step.extra_dict(extra_dict).end()
                self._step = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_evaluate_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an evaluation phase.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            state_dict = dataclasses.asdict(state)
            self._evaluate = self.trainer_process.evaluate(global_step=state.global_step, state=state_dict).begin()
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_evaluate(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._evaluate is not None:
                state_dict = dataclasses.asdict(state)
                metrics = kwargs.get("metrics", None)
                self._evaluate.extra_args(state=state_dict, metrics=metrics).end()
                self._evaluate = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_predict_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of prediction.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            state_dict = dataclasses.asdict(state)
            self._predict = self.trainer_process.predict(global_step=state.global_step, state=state_dict).begin()
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_prediction_step(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            self.trainer_process.predict_step(global_step=state.global_step, total_flos=state.total_flos)
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_predict(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._predict is not None:
                state_dict = dataclasses.asdict(state)
                self._predict.extra_args(state=state_dict, metrics=metrics).end()
                self._predict = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_save_begin(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of saving.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            state_dict = dataclasses.asdict(state)
            args_dict = args.to_dict()
            self._save = self.trainer_process.save(
                global_step=state.global_step, state=state_dict, args=args_dict
            ).begin()
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_save(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if self._save is not None:
                state_dict = dataclasses.asdict(state)
                args_dict = args.to_dict()
                self._save.extra_args(state=state_dict, args=args_dict).end()
                self._save = None
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")

    def on_log(self, args: AtorchTrainingArgs, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        if self.trainer_process is None or not state.is_world_process_zero:
            return
        try:
            if "logs" in kwargs:
                self.trainer_process.log(global_step=state.global_step, logs=kwargs["logs"])
        except Exception as e:
            self.trainer_process.error(f"Failed to report event: {e}")


def get_event_callback(target: str):
    if not is_training_event_available():
        return None

    return _TrainerEventCallback(target)
