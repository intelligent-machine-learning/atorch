import json
import time
from dataclasses import dataclass

import pytest
import torch

if torch.cuda.is_available():
    pytest.skip("Skip dynamic profile test, only run on cpu.", allow_module_level=True)

from atorch.utils.dynamic_profiler._dynamic_profile import no_exception_func  # noqa: E402
from atorch.utils.dynamic_profiler._file_monitor import (  # noqa: E402
    ThreadFileConfigMonitor,
    create_dataclass_from_dict,
    is_frozen_dataclass,
)


def test_no_exception_func():
    @no_exception_func()
    def func():
        raise Exception("test")

    assert func() is None


@dataclass(frozen=True)
class FooConfigInner:
    a: int
    b: str


@dataclass(frozen=True)
class BarConfig:
    a: int
    inner: FooConfigInner


def test_create_dataclass_from_dict():
    assert create_dataclass_from_dict({"a": 1, "inner": {"a": 2, "b": "test"}}, BarConfig) == BarConfig(
        a=1, inner=FooConfigInner(a=2, b="test")
    )

    # ignore extra fields
    assert create_dataclass_from_dict({"a": 1, "b": "test", "c": "extra"}, FooConfigInner) == FooConfigInner(
        a=1, b="test"
    )

    # missing fields
    assert create_dataclass_from_dict({"a": 1}, FooConfigInner) is None


def test_is_frozen_dataclass():
    class Foo:
        pass

    assert is_frozen_dataclass(BarConfig)
    assert not is_frozen_dataclass(Foo)

    @dataclass(frozen=False)
    class Bar:
        pass

    assert not is_frozen_dataclass(Bar)


def test_thread_file_config_monitor(tmp_path):
    config_path = tmp_path / "test_config.json"
    monitor = ThreadFileConfigMonitor(
        [config_path.as_posix()],
        BarConfig,
        poll_interval=1,
        validator=lambda x: x.a == 1,
    )
    monitor.start()

    assert monitor.get_config() is None

    with open(config_path, "w") as f:
        json.dump({"a": 1, "inner": {"a": 2, "b": "test"}}, f)

    time.sleep(2)
    assert monitor.get_config() == BarConfig(a=1, inner=FooConfigInner(a=2, b="test"))

    with open(config_path, "w") as f:
        json.dump({"a": 2, "inner": {"a": 3, "b": "test2"}}, f)

    time.sleep(2)
    # validator will return False, so config will not be updated
    assert monitor.get_config() == BarConfig(a=1, inner=FooConfigInner(a=2, b="test"))

    with open(config_path, "w") as f:
        f.write("invalid json")

    time.sleep(2)
    # file is invalid, so config will not be updated
    assert monitor.get_config() == BarConfig(a=1, inner=FooConfigInner(a=2, b="test"))

    monitor.stop()
