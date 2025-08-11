import types
import unittest
from unittest.mock import MagicMock

import torch

from atorch.utils.virtual_optimizer.patch_utils import (
    patch_chained_optimizer,
    patch_distributed_optimizer,
    virtual_distributed_optimizer_load_state_dict,
    zero_out_shard_fp32_memory,
)
from atorch.utils.virtual_optimizer.pp_calc import is_valid_pipeline_parallel_combination


class VirtualOptimizerTest(unittest.TestCase):
    def test_zero_out_shard_fp32_memory(self):
        chained_optimizer = MagicMock()
        chained_optimizer.chained_optimizers = [MagicMock()]
        chained_optimizer.chained_optimizers[0].shard_fp32_groups = [[torch.randn(4)]]
        chained_optimizer.chained_optimizers[0].shard_fp32_from_float16_groups = [[torch.randn(4)]]
        zero_out_shard_fp32_memory(chained_optimizer)

        self.assertEqual(chained_optimizer.chained_optimizers[0].shard_fp32_groups[0][0].data.shape, (1,))
        self.assertEqual(chained_optimizer.chained_optimizers[0].shard_fp32_from_float16_groups[0][0].data.shape, (1,))

    def test_patch_chained_optimizer(self):
        chained_optimizer = MagicMock()
        patch_chained_optimizer(chained_optimizer)
        self.assertTrue(hasattr(chained_optimizer, "step"))
        self.assertTrue(hasattr(chained_optimizer, "load_state_dict"))
        self.assertTrue(hasattr(chained_optimizer, "sharded_state_dict"))
        self.assertTrue(hasattr(chained_optimizer, "reload_model_params"))
        self.assertTrue(hasattr(chained_optimizer, "load_parameter_state"))

        step_result = chained_optimizer.step()
        self.assertEqual(step_result, (True, 0.0, 0))

        virtual_sharded_state_dict = chained_optimizer.sharded_state_dict(MagicMock())
        self.assertEqual(virtual_sharded_state_dict, {})

        chained_optimizer.reload_model_params()
        chained_optimizer.load_parameter_state(MagicMock())

    def test_distributed_optimizer_patch_copy_model_grads_to_main_grads(self):
        # Mock objects setup
        distributed_optimizer = MagicMock()
        distributed_optimizer.config = MagicMock()
        distributed_optimizer.config.use_precision_aware_optimizer = False
        distributed_optimizer.is_stub_optimizer = False

        distributed_optimizer.model_float16_groups = [[MagicMock()]]
        distributed_optimizer.shard_fp32_from_float16_groups = [[MagicMock()]]

        patch_distributed_optimizer(distributed_optimizer)
        self.assertTrue(hasattr(distributed_optimizer, "_copy_model_grads_to_main_grads"))

        param_range = MagicMock()
        param_range.start = 0
        param_range.end = 2
        distributed_optimizer._get_model_param_range_map = MagicMock(return_value={"param": param_range})

        model_param = torch.randn(4)
        model_param.main_grad = torch.randn(4)
        distributed_optimizer.model_float16_groups[0][0] = model_param

        shard_main_param = torch.randn(4)
        distributed_optimizer.shard_fp32_from_float16_groups[0][0] = shard_main_param

        distributed_optimizer._copy_model_grads_to_main_grads()
        self.assertEqual(
            distributed_optimizer.shard_fp32_from_float16_groups[0][0].virtual_grad.shape,
            model_param.main_grad.view(-1)[param_range.start : param_range.end].shape,
        )

    def test_distributed_optimizer_patch_get_main_grads_for_grad_norm(self):
        # Mock objects setup
        distributed_optimizer = MagicMock()
        distributed_optimizer.config = MagicMock()
        distributed_optimizer.config.use_precision_aware_optimizer = False

        patch_distributed_optimizer(distributed_optimizer)

        mock_params = []
        for _ in range(2):  # 创建两个参数作为示例
            param = torch.nn.Parameter(torch.randn(2, 2))
            # 添加 virtual_grad 属性
            param.virtual_grad = torch.randn(2, 2)
            # 添加 param_is_not_shared 需要的属性
            param.shared = False  # 不是共享参数
            # 添加 param_is_not_tensor_parallel_duplicate 需要的属性
            param.tensor_model_parallel = True  # 不是重复参数
            mock_params.append(param)

        distributed_optimizer.get_parameters = MagicMock(return_value=mock_params)
        self.assertTrue(hasattr(distributed_optimizer, "get_main_grads_for_grad_norm"))
        try:
            grads = distributed_optimizer.get_main_grads_for_grad_norm()
            self.assertIsInstance(grads, list)
            self.assertEqual(len(grads), 2)
        except Exception as e:
            print(f"megatron env may not prepared. error: {e}")

    def test_virtual_distributed_optimizer_load_state_dict(self):
        distributed_optimizer = MagicMock()

        distributed_optimizer.load_state_dict = types.MethodType(
            virtual_distributed_optimizer_load_state_dict, distributed_optimizer
        )
        distributed_optimizer.load_state_dict(MagicMock())

        self.assertTrue(hasattr(distributed_optimizer, "step"))


class UtilsTest(unittest.TestCase):
    def test_is_valid_pipeline_parallel_combination(self):
        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=12,
                pipeline_model_parallel_size=2,
                decoder_first_pipeline_num_layers=4,
                decoder_last_pipeline_num_layers=4,
                decoder_first_virtual_pipeline_num_layers=2,
                decoder_last_virtual_pipeline_num_layers=2,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=12,
                pipeline_model_parallel_size=3,
                decoder_first_pipeline_num_layers=4,
                decoder_last_pipeline_num_layers=4,
                decoder_first_virtual_pipeline_num_layers=2,
                decoder_last_virtual_pipeline_num_layers=2,
                num_virtual_stages_per_pipeline_rank=1,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=12,
                pipeline_model_parallel_size=3,
                decoder_first_pipeline_num_layers=2,
                decoder_last_pipeline_num_layers=4,
                decoder_first_virtual_pipeline_num_layers=1,
                decoder_last_virtual_pipeline_num_layers=2,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=12,
                pipeline_model_parallel_size=3,
                decoder_first_pipeline_num_layers=4,
                decoder_last_pipeline_num_layers=2,
                decoder_first_virtual_pipeline_num_layers=2,
                decoder_last_virtual_pipeline_num_layers=1,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=13,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=4,
                decoder_last_pipeline_num_layers=4,
                decoder_first_virtual_pipeline_num_layers=2,
                decoder_last_virtual_pipeline_num_layers=2,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=18,
                pipeline_model_parallel_size=6,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=3,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=1,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=18,
                pipeline_model_parallel_size=6,
                decoder_first_pipeline_num_layers=3,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=1,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=64,
                pipeline_model_parallel_size=3,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=24,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=3,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=24,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=7,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=3,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=32,
                pipeline_model_parallel_size=5,
                decoder_first_pipeline_num_layers=10,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=2,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=26,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=8,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=5,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=24,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=8,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=5,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=32,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=8,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=1,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=24,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=7,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=3,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=30,
                pipeline_model_parallel_size=5,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=10,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=2,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertFalse(
            is_valid_pipeline_parallel_combination(
                num_layers=32,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=8,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=6,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )

        self.assertTrue(
            is_valid_pipeline_parallel_combination(
                num_layers=24,
                pipeline_model_parallel_size=4,
                decoder_first_pipeline_num_layers=6,
                decoder_last_pipeline_num_layers=6,
                decoder_first_virtual_pipeline_num_layers=3,
                decoder_last_virtual_pipeline_num_layers=3,
                num_virtual_stages_per_pipeline_rank=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
