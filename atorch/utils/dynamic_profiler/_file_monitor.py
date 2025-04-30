import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar, cast

from atorch.common.log_utils import default_logger as logger

T = TypeVar("T", bound=object)


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

        # support sub-dataclass
        for k, v in filtered_dict.items():
            k_type = config_class.__dataclass_fields__[k].type  # type: ignore
            if is_frozen_dataclass(k_type):
                filtered_dict[k] = create_dataclass_from_dict(v, k_type)

        # create dataclass instance and convert to T type
        instance = config_class(**filtered_dict)  # type: ignore
        return cast(T, instance)
    except Exception as e:
        logger.error(f"Error creating config: {e}")
        return None


class ThreadFileConfigMonitor(Generic[T]):
    """
    Generic ThreadFileConfigMonitor is used to monitor file changes
    and load the config into a specified immutable dataclass type.
    """

    def __init__(
        self,
        config_path: str,
        config_class: Type[T],
        poll_interval: int = 60,
        validator: Optional[Callable[[T], bool]] = None,
    ):
        """
        Initialize the file monitor.

        Args:
            config_path: The path of the configuration file.
            config_class: The dataclass type to load the config into.
            poll_interval: The polling interval (seconds).
            validator: Optional function to validate the loaded config.
        """
        self._config_path = config_path

        if not is_frozen_dataclass(config_class):
            raise TypeError(f"{config_class} must be a frozen dataclass")

        self._config_class: Type[T] = config_class  # type: ignore
        self._poll_interval = poll_interval
        self._validator = validator
        self._last_mtime: float = 0.0
        self._current_config: Optional[T] = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

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
        logger.info(f"Start monitoring file {self._config_path}")
        while self._running:
            try:
                if self._check_file_changed():
                    logger.info(f"File {self._config_path} has changed")
                    config_dict = self._read_config()
                    if config_dict:
                        # convert dict to immutable dataclass
                        config = create_dataclass_from_dict(config_dict, self._config_class)
                        # validate config
                        if config is not None and self._is_valid_config(config):
                            with self._lock:
                                logger.info(f"Update config {config}")
                                self._current_config = config
                        else:
                            logger.warning(f"Invalid configuration found in {self._config_path}")
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

    def _check_file_changed(self) -> bool:
        """Check if the file has changed."""
        if not os.path.exists(self._config_path):
            return False

        current_mtime = os.path.getmtime(self._config_path)
        if current_mtime > self._last_mtime:
            self._last_mtime = current_mtime
            return True
        return False

    def _read_config(self) -> Dict[str, Any]:
        """Read the configuration file content."""
        try:
            with open(self._config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading config file {self._config_path}: {e}")
            return {}

    def get_config(self) -> Optional[T]:
        """
        Get the current configuration.

        Returns:
            Config object or None if no configuration has been loaded yet.
            The returned object is immutable and can be safely shared.
        """
        with self._lock:
            return self._current_config

    def set_poll_interval(self, seconds: int):
        """Set the polling interval."""
        if seconds > 0:
            self._poll_interval = seconds
