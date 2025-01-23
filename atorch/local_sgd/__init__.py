# NOTE since megatron impl is very version dependent, do not import anything megatron here
from .configs import GTAConfig, LocalSGDConfig, OuterOptimizerConfig
from .DDP import OuterOptimPeriodicModelAverager, StatefulPostLocalSGDOptimizer
from .FSDP import patch_local_sgd_to_fsdp
