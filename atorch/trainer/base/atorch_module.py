from typing import Union

from torch import nn
from transformers.modeling_utils import PreTrainedModel


class AtorchIRModel(nn.Module):
    def __init__(self, model: Union[PreTrainedModel]):
        self.origin_model = model
        self.model = self._convert_to_IR(model)

    def _convert_to_IR(self, model):
        return model
