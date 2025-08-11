def is_valid_pipeline_parallel_combination(
    num_layers,
    pipeline_model_parallel_size,
    decoder_first_pipeline_num_layers,
    decoder_last_pipeline_num_layers,
    decoder_first_virtual_pipeline_num_layers,
    decoder_last_virtual_pipeline_num_layers,
    num_virtual_stages_per_pipeline_rank,
):
    """Check if a combination of parameters is valid according to the constraints."""
    if pipeline_model_parallel_size <= 2:
        return False

    numerator = num_layers - decoder_first_pipeline_num_layers - decoder_last_pipeline_num_layers
    denominator = pipeline_model_parallel_size - 2

    if num_virtual_stages_per_pipeline_rank <= 1:
        return False

    if decoder_first_pipeline_num_layers <= num_virtual_stages_per_pipeline_rank:
        return False

    if decoder_last_pipeline_num_layers <= num_virtual_stages_per_pipeline_rank:
        return False

    if numerator % denominator != 0:
        return False

    num_layers_per_pp_stage = numerator // denominator

    if num_layers_per_pp_stage < decoder_first_pipeline_num_layers:
        return False

    if num_layers_per_pp_stage < decoder_last_pipeline_num_layers:
        return False

    if num_layers_per_pp_stage > 15:
        return False

    if decoder_first_pipeline_num_layers > num_layers_per_pp_stage:
        return False

    if decoder_last_pipeline_num_layers > num_layers_per_pp_stage:
        return False

    if num_layers_per_pp_stage % num_virtual_stages_per_pipeline_rank != 0:
        return False

    num_layers_per_vpp_stage = num_layers_per_pp_stage // num_virtual_stages_per_pipeline_rank
    decoder_first_pp_rank_num_layers_without_first_vpp_stage = (
        decoder_first_pipeline_num_layers - decoder_first_virtual_pipeline_num_layers
    )

    if num_virtual_stages_per_pipeline_rank > 1:
        if decoder_first_pp_rank_num_layers_without_first_vpp_stage % (num_virtual_stages_per_pipeline_rank - 1) != 0:
            return False
        decoder_first_pp_rank_not_first_vpp_stage_num_layers = (
            decoder_first_pp_rank_num_layers_without_first_vpp_stage // (num_virtual_stages_per_pipeline_rank - 1)
        )

        if decoder_first_pp_rank_not_first_vpp_stage_num_layers > num_layers_per_vpp_stage:
            return False
        if decoder_first_pp_rank_not_first_vpp_stage_num_layers < decoder_first_virtual_pipeline_num_layers:
            return False
    else:
        if num_layers_per_vpp_stage > decoder_first_pp_rank_num_layers_without_first_vpp_stage:
            return False

    if decoder_first_virtual_pipeline_num_layers > num_layers_per_vpp_stage:
        return False

    if num_layers_per_vpp_stage > decoder_first_virtual_pipeline_num_layers + 2:
        return False

    decoder_last_pp_rank_not_last_vpp_stage_num_layers = (
        decoder_last_pipeline_num_layers - decoder_last_virtual_pipeline_num_layers
    )

    if decoder_last_pp_rank_not_last_vpp_stage_num_layers % (num_virtual_stages_per_pipeline_rank - 1) != 0:
        return False

    decoder_last_pp_rank_not_last_vpp_stage_num_layers = decoder_last_pp_rank_not_last_vpp_stage_num_layers // (
        num_virtual_stages_per_pipeline_rank - 1
    )

    if decoder_last_pp_rank_not_last_vpp_stage_num_layers > num_layers_per_vpp_stage:
        return False

    if num_layers_per_vpp_stage > decoder_last_pp_rank_not_last_vpp_stage_num_layers + 2:
        return False

    return True
