import atexit
import functools
import json
import os
import socket
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import profiler

from atorch.common.log_utils import default_logger as logger
from atorch.common.singleton import SingletonMeta
from atorch.distributed.distributed import rank

from ._file_monitor import ThreadFileConfigMonitor

__all__ = ["init"]


def active_kineto() -> bool:
    dynolog_flag = os.getenv("KINETO_USE_DAEMON", 0)
    try:
        dynolog_flag = int(dynolog_flag)
    except ValueError:
        logger.error("Environment variable KINETO_USE_DAEMON value not valid, will be set to 0 !")
        dynolog_flag = 0

    return dynolog_flag == 1


if not active_kineto():
    if torch.__version__ >= "2.0.0":
        _origin_patch_step_function = torch.optim.Optimizer._patch_step_function
    elif torch.__version__ >= "1.8.0":
        _origin_patch_step_function = torch.optim.Optimizer._hook_for_profile  # type: ignore[attr-defined]


def no_exception_func(default_ret=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                logger.error(f"Call {func.__name__} failed. Exception: {str(ex)}")
                return default_ret
            return result

        return wrapper

    return decorator


@dataclass(frozen=True)
class ProfilerConfig:
    """Immutable profiler configuration."""

    enabled: bool = False
    output_dir: str = ""
    start_step: int = 0
    schedule_wait: int = 0
    schedule_warmup: int = 0
    schedule_active: int = 0
    schedule_repeat: int = 1
    schedule_skip_first: int = 0
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    record_shapes: bool = False
    profile_memory: bool = False
    # acc_events: bool = False # TODO: add acc_events for new version
    activities: list = field(default_factory=list)
    meta_data: dict = field(default_factory=dict)
    profile_ranks: list = field(default_factory=list)
    use_gzip: bool = True

    def __post_init__(self):
        activities = self.activities
        new_activities = []
        for activity in activities:
            prof_activity = getattr(torch.profiler.ProfilerActivity, activity.upper(), None)
            if prof_activity is None:
                logger.warning("Invalid profiler activity: %s", activity)
            else:
                new_activities.append(prof_activity)
        object.__setattr__(self, "activities", new_activities)

    def is_valid(self) -> bool:
        """Check if the configuration is valid."""
        return (
            self.enabled
            and self.schedule_active > 0
            and self.output_dir != ""
            and len(self.activities) > 0
            and len(self.profile_ranks) > 0
        )


class _DynamicProfile(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.profiler = None
        self.config: Optional[ProfilerConfig] = None
        self.step_num = 0
        self.cur_step = 0
        self._optimizer_id = 0
        self._dynamic_monitor: Optional[ThreadFileConfigMonitor[ProfilerConfig]] = None
        # hook step function
        if torch.__version__ >= "2.0.0":
            torch.optim.Optimizer._patch_step_function = _DynamicProfile.patch_step_function  # type: ignore[assignment]

        elif torch.__version__ >= "1.8.0":
            torch.optim.Optimizer._hook_for_profile = _DynamicProfile.patch_step_function  # type: ignore[attr-defined]

    def init(self, cfg_path: str):
        logger.info("init dynamic profile with cfg path: %s", cfg_path)
        self.rank = rank()
        self._dynamic_monitor = ThreadFileConfigMonitor(config_path=cfg_path, config_class=ProfilerConfig)
        self._dynamic_monitor.start()
        atexit.register(self._clean_resource)

    def _clean_resource(self):
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            logger.info("Profiler stop when process exit, check cfg json active whether over all step!")
        self._dynamic_monitor.stop()

    def step(self):
        self.cur_step += 1

        if self._dynamic_monitor is not None:
            config = self._dynamic_monitor.get_config()
            if config is not None:
                self.config = config

        if self.profiler:
            # step profiler
            self.profiler.step()
            self.step_num -= 1

            # stop profiler if step num is 0
            if 0 == self.step_num:
                self.profiler.stop()
                self.profiler = None

                logger.info("Stop Dynamic Profiler at {} step.".format(self.cur_step))
        elif self.profiler is None and self.config is not None and self.cur_step == self.config.start_step:
            # start profiler
            self.step_num = self.config.schedule_active + self.config.schedule_warmup
            self.start_profile()
            self.config = None

    def start_profile(self):
        def trace_handler():
            if self.config.profile_ranks[0] == -1 or self.rank in self.config.profile_ranks:
                return torch.profiler.tensorboard_trace_handler(
                    self.config.output_dir,
                    worker_name=f"torch_profiler_rank_{self.rank}_{socket.gethostname()}_{os.getpid()}",
                    use_gzip=self.config.use_gzip,
                )
            else:
                logger.info("Profile will not be recorded for rank %d", self.rank)

                def _dummy_writer(p):
                    # Do nothing
                    pass

                return _dummy_writer

        self.profiler = profiler.profile(
            activities=self.config.activities,
            schedule=profiler.schedule(
                wait=self.config.schedule_wait,
                warmup=self.config.schedule_warmup,
                active=self.config.schedule_active,
                repeat=self.config.schedule_repeat,
                skip_first=self.config.schedule_skip_first,
            ),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
            on_trace_ready=trace_handler(),
        )
        self.profiler.start()
        for key, value in self.config.meta_data.items():
            self.profiler.add_metadata_json(str(key), json.dumps(value))

        logger.info("Start Dynamic Profiler at {} step.".format(self.cur_step))

    @staticmethod
    def patch_step_function(optimizer: torch.optim.Optimizer):
        dp = _DynamicProfile()

        def step_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                optimizer, *_ = args
                if id(optimizer) == dp._optimizer_id:
                    dp.step()
                return out

            return wrapper

        _origin_patch_step_function(optimizer)

        # hook optimizer step function
        dp._optimizer_id = id(optimizer)
        step_hooked = getattr(optimizer.__class__.step, "step_hooked", None)
        if not step_hooked:
            optimizer.__class__.step = step_wrapper(optimizer.__class__.step)  # type: ignore[assignment]
            optimizer.__class__.step.step_hooked = True  # type: ignore[attr-defined]


# hook optimizer step function when _DynamicProfile is initialized
if not active_kineto():
    _dynamic_profile = _DynamicProfile()


@no_exception_func()
def init(path: str):
    if active_kineto():
        logger.warning("Dynamic profiler is not supported in dynolog mode, skip init.")
        return

    # init dynamic profiler
    _dynamic_profile.init(cfg_path=path)
