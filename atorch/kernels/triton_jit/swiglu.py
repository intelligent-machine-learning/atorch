from atorch.utils.import_util import is_liger_kernel_available


def liger_fused_silu(a, b):
    """Compute silu(a) * b"""
    if is_liger_kernel_available():
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction

        return LigerSiLUMulFunction.apply(a, b)
    else:
        raise RuntimeError("atorch fused silu is not available")
