from abc import ABC, abstractmethod

from transformers import TrainerCallback

from atorch.trainer.args import AtorchTrainingArgs


class CkptSaver(ABC, TrainerCallback):
    @abstractmethod
    def save(
        self, iteration: int, output_dir: str, train_args: AtorchTrainingArgs, best_model_checkpoint=None, **kwargs
    ) -> str:
        pass

    @abstractmethod
    def get_interation_path(self, output_dir: str, iteration: int, **kwargs) -> str:
        pass
