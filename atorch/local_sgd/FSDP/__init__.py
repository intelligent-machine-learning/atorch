import torch

from atorch.common.log_utils import default_logger as logger
from atorch.utils.version import torch_version  # noqa: E402


def patch_local_sgd_to_fsdp():
    if torch_version()[:2] == (2, 1):  # type: ignore
        from .torch_2_1_0 import patch_local_sgd_to_fsdp

        patch_local_sgd_to_fsdp()
    elif torch_version()[0] == 2 and torch_version()[1] >= 4:  # type: ignore
        from .torch_2_4_0 import patch_local_sgd_to_fsdp

        patch_local_sgd_to_fsdp()
    else:
        raise ValueError("Only pytorch 2.1.x and >=2.4.x supports local sgd!")
    logger.info(f"Local SGD hacked on Pytorch {torch.__version__}!")
