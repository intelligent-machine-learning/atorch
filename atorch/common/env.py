import os

from atorch.common.singleton import SingletonMeta


def parse_bool_env(name, default="False"):
    str_env = os.getenv(name, default)
    if str_env in ["1", "True", "true"]:
        return True
    elif str_env in ["0", "False", "false"]:
        return False
    else:
        raise ValueError(f"Env {name} should be True or False")


class EnvSetting(metaclass=SingletonMeta):
    # TODO: config the num in moe module which is more comprehensive
    MOE_FSDP_PREFETCH_NUM = int(os.getenv("MOE_FSDP_PREFETCH_NUM", 1))
    MOE_NPU_DISABLE_ARGSORT_REPLACE = parse_bool_env("MOE_NPU_DISABLE_ARGSORT_REPLACE")
    MOE_DISABLE_SHARED_EXPERT_OVERLAP = parse_bool_env("MOE_DISABLE_SHARED_EXPERT_OVERLAP")
    DISABLE_CHECKPOINT_PATCH = parse_bool_env("ATORCH_DISABLE_CHECKPOINT_PATCH")
    MOE_NPU_DISABLE_FUSED_KERNEL = parse_bool_env("MOE_NPU_DISABLE_FUSED_KERNEL", "True")
    MOE_NV_DISABLE_FUSED_KERNEL = parse_bool_env("MOE_NV_DISABLE_FUSED_KERNEL", "False")
    MOE_REPLACE_MINDSPEED_ALLGATHER_TOKENDISPATCHER_INDEX = parse_bool_env(
        "MOE_REPLACE_MINDSPEED_ALLGATHER_TOKENDISPATCHER_INDEX", default="True"
    )
    MOE_MLP_PREFIX = parse_bool_env("MOE_MLP_PREFIX", "True")
    DISABLE_TE_LINEAR_PATCHING = parse_bool_env("DISABLE_TE_LINEAR_PATCHING")
    DEBUG = parse_bool_env("ATORCH_DEBUG", "False")
    FORCE_FSDP2_RESHARD_AFTER_FORWARD = parse_bool_env("FORCE_FSDP2_RESHARD_AFTER_FORWARD", "False")
    CLOSE_FSDP2_BACKWARD_PREFETCH = parse_bool_env("CLOSE_FSDP2_BACKWARD_PREFETCH", "False")

    # FP8
    FORCE_QUANTIZE_PER_MICROBATCH = parse_bool_env("FORCE_QUANTIZE_PER_MICROBATCH", "False")
