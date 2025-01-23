from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Savable(ABC):
    @abstractmethod
    def save_state(self, file_path, state_dict=None, **kwargs):
        pass

    @abstractmethod
    def load_state(self, file_path, **kwargs):
        pass


class AtorchStateful(Savable):
    """
    This will be an instance of torch torch.distributed.checkpoint.stateful.Stateful, since it has the same methods
    as Stateful, although it doesn't implicitly inherit from Stateful class.
    >>> a = AtorchStateful()
    >>> print(isinstance(a, Stateful))
    ## return True
    """

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_save_filepath(self, output_dir, **kwargs):
        pass

    def save_state(self, file_path, **kwargs):
        torch.save(self.state_dict(), file_path)

    def load_state(self, file_path, **kwargs):
        self.load_state_dict(torch.load(file_path))
