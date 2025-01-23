import re

from atorch.common.log_utils import default_logger as logger

from .precision_switchable_linear import LinearPrecision, PrecisionSwitchableLinear
from .scaled_linear import ScaledLinear

try:
    import transformer_engine.pytorch as te

    te_linear = te.Linear
except (ImportError, ModuleNotFoundError):
    te_linear = None


def get_fp8_module_count(model):
    # Return 5-item tuple (total_fp8_module_count,
    # scaled_linear_module_count, scaled_linear_module_with_fp8_count,
    # precision_switchable_module_count, precision_switchable_module_with_fp8_count)
    total_fp8_module_count = 0  # total module count using fp8
    precision_switchable_module_count = 0  # PrecisionSwitchableLinear module count
    precision_switchable_module_with_fp8_count = 0  # PrecisionSwitchableLinear using fp8 module count
    scaled_linear_module_count = 0  # ScaledLinear count
    scaled_linear_module_with_fp8_count = 0  # ScaledLinear with fp8 count

    for m in model.modules():
        if te_linear is not None and isinstance(m, te_linear):
            total_fp8_module_count += 1
        elif isinstance(m, PrecisionSwitchableLinear):
            precision_switchable_module_count += 1
            if m.selection == LinearPrecision.FP8:
                total_fp8_module_count += 1
                precision_switchable_module_with_fp8_count += 1
        elif isinstance(m, ScaledLinear):
            scaled_linear_module_count += 1
            if m.use_fp8:
                total_fp8_module_count += 1
                scaled_linear_module_with_fp8_count += 1

    return (
        total_fp8_module_count,
        scaled_linear_module_count,
        scaled_linear_module_with_fp8_count,
        precision_switchable_module_count,
        precision_switchable_module_with_fp8_count,
    )


def set_linear_modules_precision(
    model,
    fp8_include_name_pattern=None,
    fp8_exclude_name_pattern=None,
    fp8_module_list=None,
    precision="fp8",
    reset_fp8_meta=True,
    verbose=False,
):
    # set precisions PrecisionSwitchableLinear modules in model
    # if fp8_module_list is not None, set modules in fp8_module_list to precision.
    # if fp8_module_list is None, set PrecisionSwitchableLinear module in model to precision if:
    #   fp8_include is None or its name matches fp8_include_name_pattern pattern AND
    #   fp8_exclude is None or its name does not match fp8_exclude_name_pattern pattern.
    # reset_fp8_meta: if reset fp8_meta when switch to fp8
    if fp8_module_list is not None:
        for m in fp8_module_list:
            if isinstance(m, PrecisionSwitchableLinear):
                m.switch_precision(precision, reset_fp8_meta=reset_fp8_meta)
            elif isinstance(m, ScaledLinear):
                m.set_use_fp8(precision == "fp8")
    else:
        set_count = 0
        for n, m in model.named_modules():
            if (fp8_include_name_pattern is None or re.search(fp8_include_name_pattern, n)) and (
                fp8_exclude_name_pattern is None or not re.search(fp8_exclude_name_pattern, n)
            ):
                if isinstance(m, PrecisionSwitchableLinear):
                    m.switch_precision(precision, reset_fp8_meta=reset_fp8_meta)
                    set_count += 1
                    if verbose:
                        logger.info(f"Set {n} precision to {precision}")
                elif isinstance(m, ScaledLinear):
                    m.set_use_fp8(precision == "fp8")
                    set_count += 1
                    if verbose:
                        logger.info(f"Set {n} precision to {precision}")
        logger.info(f"Set {set_count} modules precision to {precision}.")


def set_linear_modules_pre_cast_input_fp8_current_scaling(
    model,
    include_name_pattern=None,
    exclude_name_pattern=None,
    module_list=None,
    if_precast=True,
    verbose=False,
):
    # set precast input fp8 for PrecisionSwitchableLinear modules in model
    # if module_list is not None, set modules in module_list to if_precast.
    # if module_list is None, set PrecisionSwitchableLinear module in model to precast if:
    #   include_name_pattern is None or its name matches include_name_pattern pattern AND
    #   exclude_name_pattern is None or its name does not match exclude_name_pattern pattern.
    if module_list is not None:
        for m in module_list:
            assert isinstance(m, PrecisionSwitchableLinear)
            m.set_pre_cast_input_fp8_current_scaling(if_precast)
    else:
        count = 0
        precast_name = "current scaling" if if_precast else "not current scaling"
        for n, m in model.named_modules():
            if isinstance(m, PrecisionSwitchableLinear):
                if (include_name_pattern is None or re.search(include_name_pattern, n)) and (
                    exclude_name_pattern is None or not re.search(exclude_name_pattern, n)
                ):
                    m.set_pre_cast_input_fp8_current_scaling(if_precast)
                    count += 1
                if verbose:
                    logger.info(f"Set {n} input to {precast_name}")
        logger.info(f"PrecisionSwitchableLinear: set {count} modules precast input to {precast_name}.")
