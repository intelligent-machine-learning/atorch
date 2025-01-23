from functools import partial

from ..abstract_extension import AbstractExtension


def _permute_helper(tokens, indices, num_out_tokens: int = None, fn=None):
    if num_out_tokens is None:
        # num_out_tokens required by te
        num_out_tokens = tokens.shape[0]
    if indices.dim() == 1:
        # 2D indices required by te
        indices = indices.view(-1, 1)
    return fn(tokens, indices, num_out_tokens)


class TeMoeExtension(AbstractExtension):
    def is_available(self) -> bool:
        try:
            from transformer_engine.pytorch import moe_permute, moe_unpermute  # noqa: F401

            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def load(self):
        if not self.is_available():
            return None, None

        from transformer_engine.pytorch import moe_permute, moe_unpermute

        permute_func = partial(_permute_helper, fn=moe_permute)
        return permute_func, moe_unpermute


te_permute, te_unpermute = TeMoeExtension().load()
