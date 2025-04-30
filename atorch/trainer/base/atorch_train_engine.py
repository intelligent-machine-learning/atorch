import contextlib
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import torch.nn
from accelerate.state import GradientState
from torch.utils.data import DataLoader

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.trainer_callback import AtorchTrainerState


class AtorchTrainEngine(torch.nn.Module):
    def __init__(
        self,
        train_args,
        train_state: AtorchTrainerState,
    ):
        super().__init__()

        self.train_args: AtorchTrainingArgs = train_args
        self.gradient_state = GradientState()
        self.train_state = train_state
        self.pytorch_model = None

    @staticmethod
    def initialize(**kwargs):
        pass

    def optimizer_step(self):
        """
        Subclass should implement the following five step:
        1. unscaling gradient
        2. find inf/nan in gradients (optional)
        3. clip gradient
        4. count zero in gradients (optional)
        5. optimizer.step()
        """
        pass

    def scheduler_step(self):
        pass

    def optimizer_zero_grad(self):
        pass

    def backward(self, loss):
        pass

    @classmethod
    def build_dataloader(cls, dataset, shuffle, collate_fn, batch_size, **kwargs):
        return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, batch_size=batch_size, **kwargs)

    @abstractmethod
    def get_dataloader(self, name=None):
        pass

    @classmethod
    def from_config(cls, atorch_train_step):
        pass

    def forward(self, **batch_data):
        pass

    def train(self):
        pass

    def train_step(self, **batch_data):
        raise NotImplementedError("Subclass should implement this method.")

    def eval(self):
        pass

    def eval_step(self, **batch_data):
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: Optional[Path], best_model_checkpoint=None, **kwargs):
        """

        Args:
            output_dir:
            train_state:
            best_model_checkpoint: designed from TrainState.best_model_checkpoint, could be a folder path as str.
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def get_checkpoint_iteration_path_dir(self, output_dir: Optional[Path], **kwargs) -> Path:
        pass

    @abstractmethod
    def load_checkpoint(self, resume_from_ckpt: Optional[Path], model, optimizer=None, scheduler=None, **kwargs):
        pass

    def training_log(self, **kwargs):
        pass

    @contextlib.contextmanager
    def _no_sync(self, model):
        context = contextlib.nullcontext
        if self.use_distributed:
            context = getattr(model, "no_sync", context)

        with context():
            yield

    def _do_sync(self):
        """
        set the sync_gradients flag
        """
        if self.gradient_state.sync_with_dataloader and self.gradient_state.end_of_dataloader:
            self.train_state.steps_in_epoch = 0
            self.gradient_state._set_sync_gradients(True)
        else:
            # TODO:
            self.step += 1
            self.gradient_state._set_sync_gradients(
                (self.train_state.steps_in_epoch % self.gradient_state.num_steps) == 0
            )

    @contextlib.contextmanager
    def accumulate(self):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `torch.nn.Module`):
            # TODO
                PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
                skip gradient syncing during backward pass in distributed training
        """
        self._do_sync()

        allow_gradient_sync = self.gradient_state.sync_gradients or (
            # must sync if sync gradients need to complete an optimizer step
            # the no_sync context stops the gradients from reducing during distributed training
            # bringing speedup (potentially at some costs). Here, no_sync can be prevented
            # by setting sync_each_batch = True.
            self.train_args.use_distributed  # only relevant in distributed settings
            and self.gradient_state.plugin_kwargs.get("sync_each_batch", False)
        )
        with contextlib.ExitStack() as cm_stack:
            cm_stack.enter_context(
                contextlib.nullcontext() if allow_gradient_sync else self._no_sync(self.pytorch_model)
            )
            yield
