import math

import torch
import torch.nn as nn

try:
    from transformer_engine.pytorch import GroupedLinear, Linear

    HAS_TE = True
except (ImportError, ModuleNotFoundError):
    GroupedLinear = object
    Linear = object
    HAS_TE = False


# Toy for moe
class DummyTeGG(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_gemms):
        super().__init__()
        self.gg1 = GroupedLinear(num_gemms, hidden_size, intermediate_size)
        self.gg2 = GroupedLinear(num_gemms, intermediate_size, hidden_size)
        self.num_gemms = num_gemms
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward(self, x):
        token_num = math.prod(x.shape[:-1])
        # hack to evenly partition input x
        m_splits = [token_num // self.num_gemms] * self.num_gemms
        return self.gg2(self.gg1(x, m_splits), m_splits)


class DummyTeModel(torch.nn.Module):
    def __init__(self, hidden_size, num_gemms=4):
        super().__init__()
        self.mw1 = Linear(hidden_size, hidden_size * 2)
        self.mw2 = Linear(hidden_size * 2, hidden_size)
        self.layers = torch.nn.ModuleList(DummyTeGG(hidden_size, 2 * hidden_size, num_gemms) for _ in range(2))

    def forward(self, x):
        x = self.mw2(self.mw1(x))
        for layer in self.layers:
            x = layer(x)
        return x


def get_model(hidden_size, num_gemms=4, seed=123):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = DummyTeModel(hidden_size, num_gemms).cuda()
    return model


def get_input(batch_size, hidden_size):
    return torch.randn(batch_size, hidden_size, device=torch.device("cuda"))


def loss_func(inputs, output):
    loss = nn.MSELoss()
    return loss(inputs, output)
