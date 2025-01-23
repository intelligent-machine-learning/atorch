import inspect
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union, cast

import torch
import torch.distributed as dist
from torch.distributed._state_dict_utils import _offload_state_dict_to_cpu
from torch.distributed.checkpoint._storage_utils import _storage_setup
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.staging import AsyncStager
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.checkpoint.utils import _api_bc_check, _DistWrapper, _profile
from torch.distributed.distributed_c10d import _get_default_group

from atorch.common.log_utils import default_logger as logger


@_dcp_method_logger(log_exceptions=True)
def async_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Future:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.async_save")

    if dist.is_available() and dist.is_initialized():
        pg = process_group or _get_default_group()
        assert (
            torch.device("cpu") in pg._device_types  # type: ignore[attr-defined]
        ), "A CPU backend must be enabled for async save; try initializing process group with 'cpu:gloo,cuda:nccl'"

    storage_writer = cast(StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False))

    state_dict = _stateful_to_state_dict(state_dict)
    if isinstance(storage_writer, AsyncStager):
        staged_state_dict = storage_writer.stage(state_dict)
    else:  # provides bwc for storage_writers not implementing AsyncStager
        staged_state_dict = _offload_state_dict_to_cpu(state_dict, type_check=False)

    executor = ThreadPoolExecutor(max_workers=1)
    f: Future = executor.submit(
        save,
        staged_state_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
    )
    f.add_done_callback(lambda f: executor.shutdown(wait=False))

    if isinstance(storage_writer, AsyncStager) and storage_writer.should_synchronize_after_execute:
        storage_writer.synchronize_staging()

    return f


def _stateful_to_state_dict(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """Creates a shallow copy of `state_dict` where `state_dict` is called for each Stateful object."""
    stateful_state_dict = {}
    for key, elem in state_dict.items():
        stateful_state_dict[key] = elem.state_dict() if isinstance(elem, Stateful) else elem
    return stateful_state_dict


@_dcp_method_logger(log_exceptions=True)  # type: ignore[arg-type]
@_api_bc_check
def save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save")

    no_dist = not (dist.is_available() and dist.is_initialized())
    if no_dist:
        warnings.warn(
            "torch.distributed is unavailable or uninitialized, assuming the intent is to save in a single process."
        )

    with _profile():
        storage_writer = cast(StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False))

        return _save_state_dict(
            state_dict=_stateful_to_state_dict(state_dict),
            storage_writer=storage_writer,
            process_group=process_group,
            no_dist=no_dist,
            planner=planner,
        )


def _save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter."
            )
            planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=distW.is_coordinator,
            )
        storage_writer.set_up_storage_writer(distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        assert planner is not None
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: SavePlan = distW.reduce_scatter("plan", local_step, global_step)

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        try:
            all_writes = storage_writer.write_data(final_local_plan, planner)

            all_writes.wait()
        except Exception as e:  # Here is the difference from origin dcp state_dict_saver.py
            import traceback

            logger.error(
                f"Async saving met error during writing on rank {torch.distributed.get_rank()}, "
                f"last ckpt iteration step will not be updated, 'finish' step will NOT be executed,"
                f"for more details: {traceback.format_exc()}"
            )
            raise e

        return all_writes.value()

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint(all_results):
        assert global_metadata is not None
        storage_writer.finish(metadata=global_metadata, results=all_results)
        return global_metadata

    return distW.all_reduce("write", write_data, finish_checkpoint)
