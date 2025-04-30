from abc import ABC, abstractmethod

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.ckptloader import CkptLoader
from atorch.trainer.fsdp.fsdp_ckpt_saver import ExtraState


class FsdpCkptLoader(CkptLoader, ABC):
    @abstractmethod
    def load(  # type: ignore[override]
        self,
        resume_from_ckpt,
        model,
        train_args: AtorchTrainingArgs = None,
        optimizer=None,
        extra_state: ExtraState = None,
        ckpt_step=None,
        **kwargs
    ) -> int:
        """

        Args:
            resume_from_ckpt: checkpoint folder to load from, normally is the parent of the ckpt iteration path.
            model: the FSDP model to load state dict into, should be the same structor as the saving model.
            train_args: atorch trainer args.
            optimizer: the FSDP optimizer to load state dict into, should be the same structor as the saving optimizer.
            extra_state: the customized state to load from the ckpt
            ckpt_step: load a certain step from the ckpt root path. If None, will read the iteration mate from the
                       trace mate data file(e.g. latest_checkpointed_iteration.txt)
            **kwargs:

        Returns:
            iteration step as int

        """
        pass
