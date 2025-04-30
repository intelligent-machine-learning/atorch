from atorch.common.env import EnvSetting
from atorch.modules.moe.grouped_gemm_moe import Grouped_GEMM_MoE, MOEContext


class GroupedGEMMMoEWithMLPPrefix(Grouped_GEMM_MoE):
    def __init__(
        self,
        hidden_size,
        expert_intermediate_size,
        output_dropout_prob,
        num_experts,
        topk,
        use_swiglu=False,
        use_bias=False,
        initializer_range=0.02,
        use_expert_parallelism=False,
        expert_parallel_group=None,
        merge_w1_v1=True,
        transpose_w1=True,
        is_scale_gradient=False,
        implementation_type="MegaBlocks",
        token_dispatcher_type="AllToAll",
    ) -> None:
        assert MOEContext().MLP_PREFIX_FROM_USER_ARG is None or MOEContext().MLP_PREFIX_FROM_USER_ARG is True
        MOEContext().MLP_PREFIX_FROM_USER_ARG = True
        super().__init__(
            hidden_size,
            expert_intermediate_size,
            output_dropout_prob,
            num_experts,
            topk,
            use_swiglu,
            use_bias,
            initializer_range,
            use_expert_parallelism,
            expert_parallel_group,
            merge_w1_v1,
            transpose_w1,
            is_scale_gradient,
            implementation_type,
            token_dispatcher_type,
        )

    @property
    def w1(self):
        return self._get_w1_for_mlp_prefix()

    @property
    def w2(self):
        return self._get_w2_for_mlp_prefix()

    @property
    def v1(self):
        return self._get_v1_for_mlp_prefix()


class GroupedGEMMMoEWithoutMLPPrefix(Grouped_GEMM_MoE):
    def __init__(
        self,
        hidden_size,
        expert_intermediate_size,
        output_dropout_prob,
        num_experts,
        topk,
        use_swiglu=False,
        use_bias=False,
        initializer_range=0.02,
        use_expert_parallelism=False,
        expert_parallel_group=None,
        merge_w1_v1=True,
        transpose_w1=True,
        is_scale_gradient=False,
        implementation_type="MegaBlocks",
        token_dispatcher_type="AllToAll",
    ) -> None:
        assert MOEContext().MLP_PREFIX_FROM_USER_ARG is None or MOEContext().MLP_PREFIX_FROM_USER_ARG is False
        MOEContext().MLP_PREFIX_FROM_USER_ARG = False
        super().__init__(
            hidden_size,
            expert_intermediate_size,
            output_dropout_prob,
            num_experts,
            topk,
            use_swiglu,
            use_bias,
            initializer_range,
            use_expert_parallelism,
            expert_parallel_group,
            merge_w1_v1,
            transpose_w1,
            is_scale_gradient,
            implementation_type,
            token_dispatcher_type,
        )


def get_groupped_gemm_moe_cls(with_mlp_prefix=False):
    if with_mlp_prefix:
        if not EnvSetting().MOE_MLP_PREFIX:
            print(
                f"Env MOE_MLP_PREFIX is {EnvSetting().MOE_MLP_PREFIX}, \
                which is not same with with_mlp_prefix arg {with_mlp_prefix}, \
                the MOE_MLP_PREFIX env will be ignored."
            )
        return GroupedGEMMMoEWithMLPPrefix

    if EnvSetting().MOE_MLP_PREFIX:
        print(
            f"Env MOE_MLP_PREFIX is {EnvSetting().MOE_MLP_PREFIX}, \
            which is not same with with_mlp_prefix arg {with_mlp_prefix}, \
            the MOE_MLP_PREFIX env will be ignored."
        )

    return GroupedGEMMMoEWithoutMLPPrefix
