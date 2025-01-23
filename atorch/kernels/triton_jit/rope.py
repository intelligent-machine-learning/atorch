from atorch.utils.import_util import is_liger_kernel_available


def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if is_liger_kernel_available():
        from liger_kernel.ops.rope import LigerRopeFunction

        return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
    else:
        raise RuntimeError("liger_rotary_pos_emb is not available")
