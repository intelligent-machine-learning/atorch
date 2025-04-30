from pathlib import Path
from typing import List, Tuple

import torch
from megatron.core.dist_checkpointing.strategies.base import SaveShardedStrategy
from megatron.training.checkpointing import ensure_directory_exists

from atorch.trainer.base.dist_checkpointing.strategies.async_utils import AsyncRequest


class AsyncTorchSaveStrategy(SaveShardedStrategy):
    def __init__(self, backend: str, version: int, thread_count: int = 2):
        super().__init__(backend, version)
        self.thread_count = thread_count

    def async_save(self, save_fn_args) -> AsyncRequest:
        return self._get_save_and_finalize_callbacks(save_fn_args)

    def save(self, state_dict, file_name: Path):
        """Each async strategy can be trivially used as a sync strategy."""
        async_request = self.async_save([(state_dict, file_name)])
        async_request.execute_sync()

    def _get_save_and_finalize_callbacks(self, save_fn_args: List[Tuple]) -> AsyncRequest:
        def save_by_torch(save_fn_args):
            for save_args in save_fn_args:
                obj = save_args[0]
                file_name = save_args[1]
                if len(obj) > 0:
                    ensure_directory_exists(file_name)
                    torch.save(obj, file_name)
                else:
                    pass

        return AsyncRequest(save_by_torch, (save_fn_args,), [])

    def can_handle_sharded_objects(self):
        return False
