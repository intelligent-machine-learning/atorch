class TieWeightInfo:
    def __init__(self):
        self.tie_info = []

    def add(self, info):
        # info is a list of weights tied together.
        # info: Union[List[str], List[Tuple(int, str)]]]
        # either weight name, or (stage_id, weight_name)
        # weights in the same list are tied.
        self.tie_info.append(info)

    def num(self):
        return len(self.tie_info)

    def __getitem__(self, index):
        return self.tie_info[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]


def get_rank_stage_info(total_layer_num, pp_rank, pp_size, virtual_pp_size, style="loop", manual_stage_partition=None):
    # return current rank's stages info List[(stage_id, layer_num, starting_layer_idx)]
    # manual_stage_partition: None, List[int] or Dict[int, int]
    # style: "loop" or "v"
    total_stages = pp_size * virtual_pp_size
    stage_ids = []
    if style == "loop":
        stage_ids = [pp_rank + i * pp_size for i in range(virtual_pp_size)]
    elif style == "v":
        assert pp_size * 2 == total_stages
        stage_v_pairs = list(zip(range(pp_size), range(total_stages - 1, pp_size - 1, -1)))
        stage_ids = stage_v_pairs[pp_rank]
    else:
        assert 0, f"style={style} not implemented"

    fixed_stage_layer_num = 0
    total_fixed_stage_layers = 0
    fixed_stage_layer = {}
    if manual_stage_partition is not None:
        fixed_stage_layer_num = len(manual_stage_partition)
        if isinstance(manual_stage_partition, dict):
            total_fixed_stage_layers = sum(manual_stage_partition.values())
            for stage_idx in manual_stage_partition:
                fixed_stage_layer[stage_idx] = manual_stage_partition[stage_idx]
        else:
            assert isinstance(manual_stage_partition, (list, tuple))
            assert fixed_stage_layer_num == total_stages
            total_fixed_stage_layers = sum(manual_stage_partition)
            assert total_fixed_stage_layers == total_layer_num
            for idx, value in enumerate(manual_stage_partition):
                fixed_stage_layer[idx] = value

    if fixed_stage_layer_num < total_stages:
        per_stage_layer_num = (total_layer_num - total_fixed_stage_layers) // (total_stages - fixed_stage_layer_num)
        extra_layer_stage_num = (total_layer_num - total_fixed_stage_layers) % (total_stages - fixed_stage_layer_num)
    else:
        per_stage_layer_num = 0
        extra_layer_stage_num = 0

    layer_num_per_stage = [0] * total_stages
    for idx in range(total_stages):
        if fixed_stage_layer_num > 0 and idx in fixed_stage_layer.keys():
            layer_num_per_stage[idx] = fixed_stage_layer[idx]
        else:
            layer_num_per_stage[idx] = per_stage_layer_num
            if extra_layer_stage_num > 0:
                layer_num_per_stage[idx] += 1
                extra_layer_stage_num -= 1

    results = []
    for stage_id in stage_ids:
        layer_num = layer_num_per_stage[stage_id]
        starting_layer_idx = sum(layer_num_per_stage[:stage_id])
        results.append((stage_id, layer_num, starting_layer_idx))

    return results


def partition_model_from_model_provider(model_provider, distributed_context, config):
    pp_size = (
        distributed_context.parallel_group_size("pipe")
        if distributed_context.parallel_group_size("pipe") is not None
        else 1
    )
    pp_rank = distributed_context.parallel_rank("pipe") if distributed_context.parallel_rank("pipe") is not None else 0
    virtual_pp_size = config.virtual_pp_size if config.virtual_pp_size is not None else 1
    total_stages = pp_size * virtual_pp_size

    style = "loop" if config.partition_method == "default" else config.partition_method
    # List((stage_id, layer_num, starting_layer_idx))
    current_rank_stage_info = get_rank_stage_info(
        config.total_layer_num,
        pp_rank,
        pp_size,
        virtual_pp_size,
        style=style,
        manual_stage_partition=config.manual_stage_partition,
    )

    modules = []
    stage_ids = []

    for (stage_id, layer_num, starting_layer_idx) in current_rank_stage_info:
        pre_process = stage_id == 0
        post_process = stage_id == total_stages - 1
        module = model_provider(
            model_config=config.model_config,
            layer_num=layer_num,
            pre_process=pre_process,
            post_process=post_process,
            start_layer_idx=starting_layer_idx,
        )
        modules.append(module)
        stage_ids.append(stage_id)

    # TODO: get tie_weight_info
    tie_weight_info = None

    return modules, stage_ids, tie_weight_info


def partition_model_from_meta_model(meta_model, distributed_context, config):
    assert "Not implemented yet."
    return None, None, None
