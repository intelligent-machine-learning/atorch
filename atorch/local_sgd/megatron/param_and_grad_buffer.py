# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import os
from typing import List, Optional

import torch
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucket, _ParamAndGradBuffer, shard_buffer
from torch.distributed import _coalescing_manager

from atorch.local_sgd.configs import LocalSGDConfig

logger = logging.getLogger(__name__)


class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    Provides functionality to register when params in the bucket group have grads ready to be
    synced; an asynchronous communication call is automatically launched when _all_ params in
    the bucket group have grads ready.

    Args:
        buckets: A list of buckets.
        ddp_config: DistributedDataParallel config object.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
    """

    def __init__(
        self,
        buckets: List[_ParamAndGradBucket],
        ddp_config: DistributedDataParallelConfig,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        local_sgd_config: Optional[LocalSGDConfig] = None,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)

        # State for bookkeeping: params is the set of parameters this bucket group is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.param_to_bucket = {}
        self.params = set()
        for bucket in self.buckets:
            for param in bucket.params_list:
                self.param_to_bucket[param] = bucket
                self.params.add(param)

        self.next_param_gather_bucket_group = None

        self.reset()
        self.param_gather_handle = None
        self.param_gather_dispatched = False
        self.grad_reduce_handle = None

        self._init_local_sgd(local_sgd_config)

    def _init_local_sgd(self, local_sgd_config: Optional[LocalSGDConfig] = None):
        self.use_local_sgd = local_sgd_config is not None
        self.local_sgd_config = local_sgd_config
        if self.use_local_sgd and self.ddp_config.use_distributed_optimizer:
            logger.warning("Current implementation of Local SGD is not compatible with distributed optimizer")
            self.use_local_sgd = False

        self.iter = 0

    def reset(self):
        """
        Reset metadata in bucket group in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.is_last_microbatch = True

    def check_for_nan_in_grad(self):
        """
        Make sure norm of grads in bucket are not NaN prior to data-parallel
        all-reduce / reduce-scatter.
        """
        global_rank = torch.distributed.get_rank()
        norm_is_nan = self.buckets[0].grad_data.norm(p=2).isnan()
        for i in range(1, len(self.buckets)):
            norm_is_nan.logical_or_(self.buckets[i].grad_data.norm(p=2).isnan())
        assert not norm_is_nan, (
            f"Rank {global_rank}: found NaN in local grad norm in "
            f"backward pass before data-parallel communication collective. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # NOTE used only for distributed_optimizer, won't coexist with local sgd
    def start_param_sync(self, force_sync: bool = False):
        """
        Initiates all necessary param all-gathers for this bucket.

        When ddp_config.overlap_param_gather is set to True, dispatches an asynchronous
        communication call (unless force_sync is True). When ddp_config.overlap_param_gather
        is set to False, makes synchronous call.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings if true.
        """
        assert self.ddp_config.use_distributed_optimizer

        if force_sync:
            if self.param_gather_handle is not None:
                self.param_gather_handle.wait()
                self.param_gather_handle = None
                return
        else:
            assert self.param_gather_handle is None

        async_op = self.ddp_config.overlap_param_gather and not force_sync
        # Coalesce communication kernels across buckets in the bucket group.
        with _coalescing_manager(self.data_parallel_group, async_ops=async_op) as cm:
            for bucket in self.buckets:
                local_data_view = shard_buffer(bucket.param_data, self.data_parallel_world_size)[
                    self.data_parallel_rank
                ]
                torch.distributed._all_gather_base(
                    bucket.param_data,
                    local_data_view,
                    group=self.data_parallel_group,
                    async_op=async_op,
                )
        if async_op:
            self.param_gather_handle = cm
        else:
            # When using `_coalescing_manager`, even if a synchronous op (async_op=False) is used,
            # `cm` is not None, which is different from when `_coalescing_manager` is not used in
            # which case the torch.distributed._all_gather_base() will return None. In order to
            # maintain consistency with prior code, we need to manually set communication handle to
            # None.
            self.param_gather_handle = None
        self.param_gather_dispatched = True

    def finish_param_sync(self, skip_next_bucket_dispatch: bool = False):
        """
        Finishes param sync communication operation for this bucket. Dispatches
        next bucket's param sync if available, unless skip_next_bucket_dispatch
        is True.

        When ddp_config.overlap_param_gather is set to True, waits for asynchronous
        communication call to complete (and dispatches one if one is not already
        outstanding). Throws assertion error if ddp_config.overlap_param_gather is set to
        False.

        Args:
            skip_next_bucket_dispatch (bool, optional): if true, dispatch next
                bucket's communication if available.
        """
        assert self.ddp_config.use_distributed_optimizer
        assert self.ddp_config.overlap_param_gather

        # If current bucket's param AG has not been dispatched, dispatch it now (e.g., first
        # AG bucket in first model chunk if ddp_config.align_param_gather is False).
        if not self.param_gather_dispatched:
            self.start_param_sync()

        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            # Dispatch next bucket's asynchronous param AG.
            if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
                self.next_param_gather_bucket_group.start_param_sync()

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert self.grad_reduce_handle is None, "Should not have multiple communication calls outstanding at once"

        if self.ddp_config.check_for_nan_in_grad:
            self.check_for_nan_in_grad()

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # Use async communications only when overlap_grad_reduce is True.
        async_op = self.ddp_config.overlap_grad_reduce
        # Coalesce communication kernels across buckets in the bucket group.
        if not self.use_local_sgd or self.iter < self.local_sgd_config.local_sgd_sync_interval:
            with _coalescing_manager(self.data_parallel_group, async_ops=async_op) as cm:
                for bucket in self.buckets:
                    if self.ddp_config.use_distributed_optimizer:
                        local_data_view = shard_buffer(bucket.grad_data, self.data_parallel_world_size)[
                            self.data_parallel_rank
                        ]
                        torch.distributed._reduce_scatter_base(
                            local_data_view,
                            bucket.grad_data,
                            op=reduce_op,
                            group=self.data_parallel_group,
                            async_op=async_op,
                        )
                    else:
                        torch.distributed.all_reduce(
                            bucket.grad_data,
                            op=reduce_op,
                            group=self.data_parallel_group,
                            async_op=async_op,
                        )
            if async_op:
                self.grad_reduce_handle = cm
            else:
                # When using `_coalescing_manager`, even if a synchronous op (async_op=False) is used,
                # `cm` is not None, which is different from when `_coalescing_manager` is not used in
                # which case the torch.distributed._reduce_scatter_base() will return None. In order to
                # maintain consistency with prior code, we need to manually set communication handle to
                # None.
                self.grad_reduce_handle = None
        else:
            # self.grad_reduce_handle = torch.futures.Future()
            # # This is just to mark the reduce_handle as completed
            # # since nowhere in code make use of the wait result of grad_reduce_handle
            # # I suppose it's safe to set it a random result
            # # TODO verify this is true
            # self.grad_reduce_handle.set_result(42)
            self.grad_reduce_handle = None  # follow the standard way, just skip it

        if self.use_local_sgd and self.iter < self.local_sgd_config.local_sgd_warmup_steps:
            self.iter += 1

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, waits for asynchronous
        communication call to complete. When ddp_config.overlap_grad_reduce is set to False,
        makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        self.param_gather_dispatched = False
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.grad_reduce_handle is not None, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.params_with_grad)}/{len(self.params)} params have grad available)"
        )
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and ddp_config.overlap_grad_reduce
        is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            assert param in self.param_to_bucket, "Param is not in the bucket group"
            assert param not in self.params_with_grad, "Cannot set grad twice"
            self.params_with_grad.add(param)
            # If all params in bucket group have grads available, issue communication call.
            if len(self.params_with_grad) == len(self.params):
                self.start_grad_sync()


def partition_buckets(
    buffers: List[_ParamAndGradBuffer],
    force_single_bucket_group: bool = False,
    local_sgd_config: Optional[LocalSGDConfig] = None,
) -> List[_ParamAndGradBucketGroup]:
    """
    Automatically regroup the buckets of input buffers and return a list of bucket groups.

    In some scenarios, we need to put buckets from different buffers into a group so that their
    communication can be aggregated.

    For example, when there are both fp8 weights and bf16 biases in the model and virtual
    pipeline parallelism is enabled, each model chunk will have an fp8 bucket and a bf16 bucket,
    which doubles the number of communication kernels, and because of the use of
    CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back communications will prevent the
    overlap of communication kernels with computation kernels.

    The grouping strategy is:
    1. If force_single_bucket_group is True, put all buckets across all buffers into a single
       bucket group.
    2. If force_single_bucket_group is False, when there is no fp8 buffer in the input buffers,
       let each bucket group have only one bucket.
    3. If force_single_bucket_group is False, when using fp8 params, merge all non-fp8 buckets
       into the last fp8 bucket group.
       - Since the non-fp8 parameters (typically the biases of various layers) are relatively
         small, they are likely to be grouped into a single non-fp8 bucket.
       - The fp8 buckets start from the end of the model, i.e., the first bucket corresponds to
         the end of the model, while the last bucket corresponds to the beginning.
       - If we combine the non-fp8 bucket with the first fp8 bucket, we cannot initiate the
         reduce-scatter to synchronize gradients after the backward pass at the end of the model
         has completed. This is because we need to wait for the non-fp8 params from the beginning
         layers to obtain their gradients.
       - Combining the non-fp8 bucket with the last fp8 bucket can help avoid this issue.

    Args:
        buffers (list): list of input buffers.
        single_bucket_group_per_buffer (bool, optional): force group all buckets in each buffer
            into a single bucket group.
    """

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        # Make sure that the param_dtype of any two buffers is different.
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    # Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _ParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size, local_sgd_config=local_sgd_config
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        # Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have
        #         only one bucket.
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                        local_sgd_config=local_sgd_config,
                    )
                )
        return bucket_groups
    else:
        # Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                # The last bucket group.
                group_buckets = [bucket] + non_fp8_buckets
            else:
                # The first N-1 bucket groups.
                group_buckets = [bucket]
            bucket_groups.append(
                _ParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                    local_sgd_config=local_sgd_config,
                )
            )
        return bucket_groups
