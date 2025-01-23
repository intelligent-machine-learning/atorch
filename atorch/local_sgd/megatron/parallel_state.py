# NOQA
import os
from datetime import timedelta
from functools import partial
from itertools import cycle
from typing import Callable, List, Optional

import megatron.core.parallel_state as parallel_state
import torch
from megatron.core.parallel_state import (
    RankGenerator,
    _set_global_memory_buffer,
    default_embedding_ranks,
    default_position_embedding_ranks,
    get_context_parallel_world_size,
    get_data_parallel_group,
    get_nccl_options,
)

# The orthogonal group to DP group, In local SGD mode, sync on this group instead of the world
_NON_DATA_PARALLEL_GROUP = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: Optional[int] = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
) -> None:
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if encoder_tensor_model_parallel_size == 0 and encoder_pipeline_model_parallel_size > 0:
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank)

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if encoder_pipeline_model_parallel_size > 0:
        parallel_state._PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if encoder_tensor_model_parallel_size > 0:
        assert encoder_pipeline_model_parallel_size > 0
        assert (
            encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ), "We do not support encoders with more TP than the decoder."

    encoder_model_size = (
        encoder_tensor_model_parallel_size * encoder_pipeline_model_parallel_size * context_parallel_size
    )
    decoder_model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(f"world_size ({world_size}) is not divisible by {total_model_size}")

    data_parallel_size: int = world_size // total_model_size

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by " "expert_model_parallel_size "
        )

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    assert (
        encoder_world_size + decoder_world_size == world_size
    ), f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}"

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError("pipeline-model-parallel size should be greater than 1 with interleaved schedule")
        parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        parallel_state._PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml  # type: ignore
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs " "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    def generator_wrapper(group_type, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)
        if encoder_rank_generator is None:
            if rank == 0:
                print(f"group type: {group_type}; ranks: {d_ranks}")
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == "pp":
            # Map 1 encoder tp rank to several decoder tp ranks, because
            # these won't be the same size.
            for x, y in zip(cycle(e_ranks), d_ranks):
                yield x + y
        elif group_type == "tp-pp":
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            assert len(e_ranks) == len(d_ranks)
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _NON_DATA_PARALLEL_GROUP
    assert parallel_state._DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"

    dp_ranks = []
    for ranks in generator_wrapper("dp"):
        dp_ranks.append(ranks)
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options("dp", nccl_comm_cfgs))
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            parallel_state._DATA_PARALLEL_GROUP = group
            parallel_state._DATA_PARALLEL_GROUP_GLOO = group_gloo
            parallel_state._DATA_PARALLEL_GLOBAL_RANKS = ranks

    if len(dp_ranks) > 1:
        non_dp_ranks = list(zip(*dp_ranks))
        for ranks in non_dp_ranks:
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options("dp", nccl_comm_cfgs)
            )
            if rank in ranks:
                _NON_DATA_PARALLEL_GROUP = group

    for ranks_with_cp in generator_wrapper("dp-cp"):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, timeout=timeout, backend="gloo")
        if rank in ranks_with_cp:
            parallel_state._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    assert parallel_state._CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    for ranks in generator_wrapper("cp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options("cp", nccl_comm_cfgs))
        if rank in ranks:
            parallel_state._CONTEXT_PARALLEL_GROUP = group
            parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    assert parallel_state._MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for ranks in generator_wrapper("tp-pp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options("mp", nccl_comm_cfgs))
        if rank in ranks:
            parallel_state._MODEL_PARALLEL_GROUP = group

    # Build the model-parallel groups with expert parallel
    assert (
        parallel_state._MODEL_AND_EXPERT_PARALLEL_GROUP is None
    ), "model and expert parallel group is already initialized"
    for ranks in generator_wrapper("tp-ep-pp", independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("mp_exp", nccl_comm_cfgs)
        )
        if rank in ranks:
            parallel_state._MODEL_AND_EXPERT_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    assert parallel_state._TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    for ranks in generator_wrapper("tp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options("tp", nccl_comm_cfgs))
        if rank in ranks:
            parallel_state._TENSOR_MODEL_PARALLEL_GROUP = group
            parallel_state._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    assert parallel_state._PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    assert parallel_state._EMBEDDING_GROUP is None, "embedding group is already initialized"
    assert parallel_state._POSITION_EMBEDDING_GROUP is None, "position embedding group is already initialized"
    for ranks in generator_wrapper("pp"):
        group = torch.distributed.new_group(ranks, timeout=timeout, pg_options=get_nccl_options("pp", nccl_comm_cfgs))
        if rank in ranks:
            if parallel_state._PIPELINE_MODEL_PARALLEL_GROUP is None:
                parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = group
                parallel_state._PIPELINE_GLOBAL_RANKS = ranks
            elif isinstance(parallel_state._PIPELINE_GLOBAL_RANKS[0], list):
                parallel_state._PIPELINE_MODEL_PARALLEL_GROUP.append(group)
                parallel_state._PIPELINE_GLOBAL_RANKS.append(ranks)
            else:
                parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = [parallel_state._PIPELINE_MODEL_PARALLEL_GROUP, group]
                parallel_state._PIPELINE_GLOBAL_RANKS = [parallel_state._PIPELINE_GLOBAL_RANKS, ranks]

        embedding_ranks = get_embedding_ranks(ranks)  # type: ignore # NOQA
        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options("embd", nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            parallel_state._EMBEDDING_GROUP = group
            parallel_state._EMBEDDING_GLOBAL_RANKS = embedding_ranks

        position_embedding_ranks = get_position_embedding_ranks(ranks)  # type: ignore # NOQA
        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options("embd", nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            parallel_state._POSITION_EMBEDDING_GROUP = group
            parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    assert parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP is None, "Tensor + data parallel group is already initialized"
    for ranks in generator_wrapper("tp-dp-cp"):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("tp_dp_cp", nccl_comm_cfgs)
        )
        if rank in ranks:
            parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in generator_wrapper("tp-dp"):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("tp_dp", nccl_comm_cfgs)
        )
        if rank in ranks:
            parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = group

    assert (
        parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP is None
    ), "Tensor + context parallel group is already initialized"
    for ranks in generator_wrapper("tp-cp"):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("tp_cp", nccl_comm_cfgs)
        )
        if rank in ranks:
            parallel_state._TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    assert parallel_state._EXPERT_MODEL_PARALLEL_GROUP is None, "Expert parallel group is already initialized"
    assert (
        parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), "Tensor + expert parallel group is already initialized"
    assert parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP is None, "Data modulo expert group is already initialized"
    assert (
        parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP is None
    ), "Data modulo expert group with context parallel is already initialized"

    for ranks in generator_wrapper("tp-ep", independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("tp_exp", nccl_comm_cfgs)
        )
        if rank in ranks:
            parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

    for ranks in generator_wrapper("ep", independent_ep=True):
        group = torch.distributed.new_group(ranks, pg_options=get_nccl_options("exp", nccl_comm_cfgs))
        if rank in ranks:
            parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

    for ranks in generator_wrapper("dp", independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options("dp_modulo_exp", nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
            parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

    for ranks in generator_wrapper("dp-cp", independent_ep=True):
        # Lazy initialization of the group
        if get_context_parallel_world_size() > 1:
            group = torch.distributed.new_group(
                ranks,
                timeout=timeout,
                pg_options=get_nccl_options("dp_modulo_exp_cp", nccl_comm_cfgs),
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        else:
            group_gloo = parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
        if rank in ranks:
            parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP = group
            parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()


def get_non_data_parallel_group():
    return _NON_DATA_PARALLEL_GROUP
