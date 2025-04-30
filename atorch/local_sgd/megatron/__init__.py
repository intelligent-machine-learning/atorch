from packaging import version

from atorch.common.log_utils import default_logger as logger
from atorch.utils.import_util import is_megatron_lm_available
from atorch.utils.version import get_megatron_version, is_megatron_version_bigger_than

if is_megatron_lm_available():
    import megatron
    import megatron.training
    import megatron.training.arguments
    from megatron import core as megatron_core
    from megatron.core.package_info import __version__ as megatron_version

    if is_megatron_version_bigger_than("0.9.0"):
        # these also imports megatron
        from .optimizer import get_megatron_optimizer
        from .parallel_state import initialize_model_parallel
        from .timers import Timers as LSDTimers
        from .training import get_model
    else:
        logger.info(f"Local SGD is not supported on Megatron {get_megatron_version()}")

from .arguments import local_sgd_args_provider


def _set_lsd_timers(args):
    from megatron.training.global_vars import _ensure_var_is_not_initialized

    _ensure_var_is_not_initialized(megatron.training.global_vars._GLOBAL_TIMERS, "timers")
    megatron.training.global_vars._GLOBAL_TIMERS = LSDTimers(args.timing_log_level, args.timing_log_option)


def patch_megatron_for_local_sgd():
    logger.warning("Local SGD Megatron patches must be applied before megatron is initialized")
    if is_megatron_lm_available() and is_megatron_version_bigger_than("0.9.0"):
        # patch the Timers
        megatron_core.Timers = LSDTimers
        # patch initializer
        megatron_core.parallel_state.initialize_model_parallel = initialize_model_parallel
        megatron.training.training.get_model = get_model
        megatron.training.global_vars._set_timers = _set_lsd_timers
