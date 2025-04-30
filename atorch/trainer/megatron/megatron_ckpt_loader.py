from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import torch
from megatron.training import get_args

from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.base.ckptloader import CkptLoader


class MegatronCkptLoader(CkptLoader):
    @abstractmethod
    def load(  # type: ignore[override]
        self,
        resume_from_ckpt: Path = None,
        model=None,
        optimizer=None,
        scheduler=None,
        train_args: AtorchTrainingArgs = None,
    ) -> Tuple[int, int]:
        pass


class MegatronOriginSyncLoader(MegatronCkptLoader):
    def load(  # type: ignore[override]
        self,
        resume_from_ckpt: Path = None,
        model=None,
        optimizer=None,
        scheduler=None,
        train_args: AtorchTrainingArgs = None,
    ):
        assert model is not None, "Megatron load model should not be None"
        assert optimizer is not None, "Megatron load optimizer should not be None"
        assert scheduler is not None, "Megatron load scheduler should not be None"

        if resume_from_ckpt is not None:
            megatron_args = get_args()
            if isinstance(resume_from_ckpt, Path):
                megatron_args.load = str(resume_from_ckpt)
            else:  # normally is str
                megatron_args.load = resume_from_ckpt

        torch.distributed.barrier()

        from megatron.training.checkpointing import load_checkpoint

        iteration, num_floating_point_operations_so_far = load_checkpoint(model, optimizer, scheduler)

        torch.distributed.barrier()

        return iteration, num_floating_point_operations_so_far
