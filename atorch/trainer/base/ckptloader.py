from abc import ABC, abstractmethod


class CkptLoader(ABC):
    @abstractmethod
    def load(self, resume_from_ckpt: str, model, **kwargs):
        pass
