import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import field, is_dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, cast

from atorch.common.log_utils import default_logger as logger

T = TypeVar("T", bound=object)
_UPDATE_TIME_KEY = "x_config_update_time"


def is_frozen_dataclass(config_class: Type[T]) -> bool:
    if not isinstance(config_class, type):
        return False

    if not is_dataclass(config_class):
        return False

    # check if the dataclass is frozen
    params = getattr(config_class, "__dataclass_params__", None)
    if params is not None:
        return params.frozen

    return True


def datetime_field(date_format: str, default: Optional[datetime] = None, **kwargs) -> Any:
    """Create a dataclass field for datetime objects with a specific format."""
    return field(default=default, metadata={"is_datetime": True, "format": date_format}, **kwargs)


def create_dataclass_from_dict(config_dict: Dict[str, Any], config_class: Type[T]) -> Optional[T]:
    """Convert a dictionary to the specified config class instance."""
    try:
        # get all fields of the dataclass
        fields = {f.name for f in config_class.__dataclass_fields__.values()}  # type: ignore
        filtered_dict = {k: v for k, v in config_dict.items() if k in fields}

        # handle nested mutable objects
        for k, v in filtered_dict.items():
            if isinstance(v, dict):
                filtered_dict[k] = deepcopy(v)
            elif isinstance(v, list):
                filtered_dict[k] = tuple(deepcopy(v))

        for k, v in filtered_dict.items():
            k_field = config_class.__dataclass_fields__[k]  # type: ignore

            # support sub-dataclass
            if is_frozen_dataclass(k_field.type):
                filtered_dict[k] = create_dataclass_from_dict(v, k_field.type)

            # support datetime
            if k_field.metadata.get("is_datetime", False):
                if isinstance(v, str):
                    filtered_dict[k] = datetime.strptime(v, k_field.metadata["format"])
                elif isinstance(v, datetime):
                    filtered_dict[k] = v
                else:
                    raise ValueError(f"Invalid datetime value: {v}")

        # create dataclass instance and convert to T type
        instance = config_class(**filtered_dict)  # type: ignore
        return cast(T, instance)
    except Exception as e:
        logger.error(f"Error creating config: {e}")
        return None


class FileWatcher:
    def __init__(self, file_path: str, expire_time: int = 600):
        self._file_path = file_path
        self._expire_time = expire_time
        self._last_mtime = 0.0

    def has_changed(self) -> bool:
        """Check if the file has changed and not expired."""
        if not os.path.exists(self._file_path):
            return False

        current_mtime = os.path.getmtime(self._file_path)
        if current_mtime > self._last_mtime:
            self._last_mtime = current_mtime
            # if expire time is set, check if the file is expired
            if self._expire_time > 0:
                if current_mtime + self._expire_time < time.time():
                    return False
            return True
        return False

    def read_if_modified(self) -> Tuple[Optional[str], Optional[datetime]]:
        """Read the file if it has been modified."""
        if self.has_changed():
            logger.info(f"File {self._file_path} has been modified within {self._expire_time} seconds")
            with open(self._file_path, "r") as f:
                return f.read(), datetime.fromtimestamp(self._last_mtime)
        return None, None

    def __str__(self) -> str:
        return f"FileWatcher(file_path={self._file_path}, expire_time={self._expire_time})"


class ThreadFileConfigMonitor(Generic[T]):
    """
    Generic ThreadFileConfigMonitor is used to monitor file changes
    and load the config into a specified immutable dataclass type.

    if all config files are modified, the last one will be used.
    """

    def __init__(
        self,
        config_paths: List[str],
        config_class: Type[T],
        poll_interval: int = 60,
        expire_time: int = 600,
        validator: Optional[Callable[[T], bool]] = None,
    ):
        """
        Initialize the file monitor.

        Args:
            config_paths: The path of the configuration files.
            config_class: The dataclass type to load the config into.
            poll_interval: The polling interval (seconds).
            validator: Optional function to validate the loaded config.
        """
        self._file_monitors = [FileWatcher(path, expire_time) for path in config_paths]
        if not is_frozen_dataclass(config_class):
            raise TypeError(f"{config_class} must be a frozen dataclass")

        self._config_class: Type[T] = config_class  # type: ignore
        self._poll_interval = poll_interval
        self._expire_time = expire_time
        self._validator = validator
        self._last_mtime: float = 0.0
        self._current_config: Optional[T] = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._check_config_if_modified = False

    def start(self):
        """Start the monitor thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the monitor thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Monitor loop, check file changes."""
        logger.info("Start monitoring files: %s", self._file_monitors)
        while self._running:
            try:
                for file_monitor in self._file_monitors:
                    content, last_mtime = file_monitor.read_if_modified()
                    if content:
                        config_dict = json.loads(content)
                        if config_dict:
                            # inject update time
                            config_dict[_UPDATE_TIME_KEY] = last_mtime
                            # convert dict to immutable dataclass
                            config = create_dataclass_from_dict(config_dict, self._config_class)
                            # validate config
                            if config is not None and self._is_valid_config(config):
                                with self._lock:
                                    logger.info(f"Update config {config}")
                                    self._current_config = config
                                    self._check_config_if_modified = True
            except Exception as e:
                logger.error(f"Error in file monitor: {e}")

            time.sleep(self._poll_interval)

    def _is_valid_config(self, config: T) -> bool:
        """Validate the configuration."""
        if self._validator:
            return self._validator(config)

        if hasattr(config, "is_valid") and callable(getattr(config, "is_valid")):
            return config.is_valid()  # type: ignore

        return True

    def get_config(self) -> Optional[T]:
        """
        Get the current configuration.

        Returns:
            Config object or None if no configuration has been loaded yet.
            The returned object is immutable and can be safely shared.
        """
        with self._lock:
            return self._current_config

    def get_config_if_modified(self) -> Optional[T]:
        """
        Get the current configuration if it has been modified after last call.
        """
        with self._lock:
            if self._current_config is not None and self._check_config_if_modified:
                self._check_config_if_modified = False
                return self._current_config
            return None

    def set_poll_interval(self, seconds: int):
        """Set the polling interval."""
        if seconds > 0:
            self._poll_interval = seconds
