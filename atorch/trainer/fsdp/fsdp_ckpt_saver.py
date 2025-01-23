import os.path
from abc import ABC, abstractmethod

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.ckptsaver import CkptSaver
from atorch.trainer.base.inferface import AtorchStateful


def get_ckpt_trace_file_path(checkpoint_root_dir):
    return os.path.join(checkpoint_root_dir, "latest_checkpointed_iteration.txt")


class ExtraState(AtorchStateful):
    def __init__(self, extra_state_dict=None):
        if extra_state_dict is None:
            self.extra_state_dict = {}
        else:
            self.extra_state_dict = extra_state_dict

    def get_save_filepath(self, output_dir, **kwargs):
        return os.path.join(output_dir, "extra_state.pth")

    def state_dict(self):
        return self.extra_state_dict

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        self.extra_state_dict = state_dict


class FsdpCkptSaver(CkptSaver, ABC):
    def get_interation_path(self, output_dir: str, iteration: int, **kwargs):
        sub_iteration_folder = "iter_{:07d}".format(iteration)
        return os.path.join(output_dir, sub_iteration_folder)

    @abstractmethod
    def save(  # type: ignore[override]
        self,
        iteration: int,
        output_dir: str = None,
        train_args: AtorchTrainingArgs = None,
        best_model_checkpoint=None,
        module=None,
        optimizer=None,
        extra_state: ExtraState = None,
    ) -> str:
        """

        Args:
            iteration: the checkpoint step to save.
            output_dir: the checkpoint output dir, normally is the parent of the iteration ckpt path.
            train_args: atorch training args, with a lot of custom settings which may infer the saving method, depends
                        on the function implement.
            best_model_checkpoint: the best model checkpoint path. Like: /output_dir/iter_00015000. When set, normally
                        this ckpt should not be deleted by ckpt rotation function.
            module: FSDP model to save
            optimizer: the optimizer corresponding the model to save
            extra_state: some customized metadata to save, could be load when loading the ckpt

        Returns:
            the real output_dir to save (normally is the parent of the iteration ckpt path)

        """
        pass
