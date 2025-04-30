import os
import shutil
import socket
import tempfile
from contextlib import nullcontext

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.trainer.args import AtorchTrainingArgs
from atorch.utils.dynamic_profiler import init


def get_profiler(args: AtorchTrainingArgs):
    profiler_type = args.profiler_type
    profiler_file_path = args.profiler_file_path
    profiler_config = args.profiler_config

    if profiler_type is None or profiler_type == "nsys":
        return nullcontext()
    elif profiler_type == "hw":
        import torch_npu

        try:
            from torch_npu.profiler.profiler_path_creator import ProfPathCreator
        except (ImportError, ModuleNotFoundError):
            from torch_npu.profiler._profiler_path_creator import ProfPathCreator

        if profiler_file_path is not None and profiler_type is not None:
            os.makedirs(profiler_file_path, exist_ok=True)

        def gen_temp_dir():
            npu_tmpdir = tempfile.mkdtemp()
            logger.info(f"NPU profiler tempdir {npu_tmpdir} generated.")
            return npu_tmpdir

        def move_file_to_user_path(file_path, user_input_path):
            try:
                if not os.path.exists(user_input_path):
                    os.makedirs(user_input_path)
                for item in os.listdir(file_path):
                    source_item = os.path.join(file_path, item)
                    destination_item = os.path.join(user_input_path, item)
                    shutil.move(source_item, destination_item)
                os.rmdir(file_path)
                return True
            except Exception as e:
                logger.exception(str(e))
                return False

        def udf_tensorboard_trace_handler(
            profiler_dir_path, local_temp_path: str = None, worker_name: str = None, analyse_flag: bool = True
        ):
            ProfPathCreator().init(worker_name=worker_name, dir_name=local_temp_path)

            def handler_fn(prof_inst) -> None:
                if analyse_flag:
                    prof_inst.analyse()
                result = move_file_to_user_path(local_temp_path, profiler_dir_path)
                if result is True:
                    logger.info(f"Local temp file has been moved to {profiler_dir_path} successfully.")
                else:
                    logger.warning(f"Error occurred when moving local temp file to {profiler_dir_path}.")

            return handler_fn

        temp_dir_path = gen_temp_dir()
        default_profiler_config = dict(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch_npu.profiler.schedule(
                wait=args.profiler_schedule_wait,
                warmup=args.profiler_schedule_warmup,
                active=args.profiler_schedule_active,
                repeat=args.profiler_schedule_repeat,
                skip_first=args.profiler_schedule_skip_first,
            ),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
            ),
            on_trace_ready=udf_tensorboard_trace_handler(profiler_file_path, temp_dir_path),
        )

        default_profiler_config.update(profiler_config)

        return torch_npu.profiler.profile(**default_profiler_config)

    elif profiler_type == "hw_dp":
        from torch_npu.profiler import dynamic_profile as dp

        assert (
            args.dynamic_profiler_config_path is not None
        ), "Please set the arg 'dynamic_profiler_config_path' when using torch npu dynamic profiler."

        dp.init(args.dynamic_profiler_config_path)

        return nullcontext()

    elif profiler_type == "nv":
        import torch
        import torch.nn
        import torch.optim
        import torch.profiler
        import torch.utils.data

        if profiler_file_path is not None and profiler_type is not None:
            os.makedirs(profiler_file_path, exist_ok=True)

        rank = atorch.rank()

        def trace_handler():
            if args.profile_ranks == [-1] or rank in args.profile_ranks:
                return torch.profiler.tensorboard_trace_handler(
                    profiler_file_path,
                    worker_name=f"torch_profiler_rank{rank}_{socket.gethostname()}_{os.getpid()}",
                    use_gzip=args.profile_use_gzip,
                )
            else:

                def _dummy_writer(p):
                    # Do nothing
                    pass

                return _dummy_writer

        default_profiler_config = dict(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch.profiler.schedule(
                wait=args.profiler_schedule_wait,
                warmup=args.profiler_schedule_warmup,
                active=args.profiler_schedule_active,
                repeat=args.profiler_schedule_repeat,
                skip_first=args.profiler_schedule_skip_first,
            ),
            on_trace_ready=trace_handler(),
        )
        default_profiler_config.update(profiler_config)
        return torch.profiler.profile(**default_profiler_config)

    elif profiler_type == "nv_dp":
        assert (
            args.dynamic_profiler_config_path is not None
        ), "Please set the arg 'dynamic_profiler_config_path' when using nv dynamic profiler."

        try:
            init(args.dynamic_profiler_config_path)
        except Exception as e:
            logger.warning(f"Failed to initialize dynamic profiler: {e}")

        return nullcontext()

    else:
        logger.warning(
            f"Unsupported profiler_type:{profiler_type}. Please use one of ['hw', 'hw_dp', 'nv', 'nv_dp', 'nsys']."
        )
        return nullcontext()
