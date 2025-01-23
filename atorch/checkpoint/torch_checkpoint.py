# Based on torchtitian checkpoint codes.
import functools
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from atorch.common.log_utils import default_logger as logger
from atorch.utils.config import Config
from atorch.utils.version import torch_version

if torch_version() >= (2, 4, 0):  # type: ignore
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.checkpoint.stateful import Stateful
else:
    dcp = None
    StateDictOptions = None
    get_model_state_dict = None
    get_optimizer_state_dict = None
    set_model_state_dict = None
    set_optimizer_state_dict = None
    Stateful = object


@dataclass
class TrainState(Stateful):
    step: int = 0

    def state_dict(self):
        return {"step": torch.tensor(self.step, dtype=torch.int32)}

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"].item()


class ModelStateWrapper(Stateful):
    def __init__(self, model: Union[nn.Module, List[nn.Module]]):
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self):
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class OptimizerStateWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ):
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> Dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class TorchDCPCheckpointManager:
    def __init__(
        self,
        config: Dict,
        model_parts: List[nn.Module],
        optimizers: List[torch.optim.Optimizer],
        lr_schedulers: List[Any],
        states: Dict[str, Any],
        dataloader: Optional[DataLoader] = None,
    ):
        assert len(model_parts) == len(optimizers)
        assert len(model_parts) == len(lr_schedulers)

        self.async_save = False
        self.config = Config(config)
        self.states = states
        self.save_load_folder = self.config.save_load_folder

        self.states.update(
            {
                "model": ModelStateWrapper(model_parts),
                "optimizer": OptimizerStateWrapper(model_parts, optimizers),
                "dataloader": dataloader,
            }
        )

        if len(lr_schedulers) == 1:
            self.states["lr_scheduler"] = lr_schedulers[0]
        else:
            # For now, pipeline-parallel with looped schedules does not support resharding for lr_scheduler.
            # It should only support saving and loading a distributed checkpoint with the same number of pp ranks
            for idx, lr_scheduler in enumerate(lr_schedulers):
                self.states[f"lr_scheduler_{idx}"] = lr_scheduler

    def _gen_checkpoint_id(self, step: int):
        return os.path.join(self.save_load_folder, f"step-{step}")

    def save(self, cur_step: int):
        start_time = time.monotonic()
        checkpoint_id = self._gen_checkpoint_id(cur_step)

        if self.async_save:
            raise NotImplementedError()
        else:
            dcp.save(self.states, checkpoint_id=checkpoint_id)

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled)"
            f"in {time.monotonic() - start_time:.2f} seconds."
        )

    def _search_step_num(self):
        step_counts = []
        for filename in os.listdir(self.save_load_folder):
            match = re.search(r"step-(\d+)", filename)
            metadata_probe = os.path.join(self.save_load_folder, filename, ".metadata")
            if match and os.path.isfile(metadata_probe):
                step_counts.append(int(match.group(1)))

        assert len(step_counts) > 0, "no step args and not found useful step folder in save_load_folder configured."

        return max(step_counts)

    def load(self, step: int = -1):
        if not os.path.exists(self.save_load_folder) or not os.path.isdir(self.save_load_folder):
            return False

        if step == -1:
            step = self._search_step_num()

        original_stateful_states = {k: v for k, v in self.states.items() if isinstance(v, Stateful)}

        logger.info(f"Loading the checkpoint at step {step}.")
        start_time = time.monotonic()
        dcp.load(
            self.states,
            checkpoint_id=self._gen_checkpoint_id(step),
        )
        logger.info(f"Finished loading the checkpoint in {time.monotonic() - start_time:.2f} seconds.")
        # bugfix from above: restore the original stateful objects,
        # whose states were already updated in-place by dcp.load()
        self.states.update(original_stateful_states)
