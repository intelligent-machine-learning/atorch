from .precision_switchable_linear import LinearPrecision, PrecisionSwitchableLinear
from .scaled_linear import ScaledLinear, fp8_valid_shape_check
from .utils import (
    get_fp8_module_count,
    set_linear_modules_pre_cast_input_fp8_current_scaling,
    set_linear_modules_precision,
)
