from abc import ABC, abstractmethod

from atorch.trainer.args import AtorchTrainingArgs


class CkptLoader(ABC):
    @abstractmethod
    def load(self, resume_from_ckpt: str, model, train_args: AtorchTrainingArgs = None, **kwargs):
        pass
