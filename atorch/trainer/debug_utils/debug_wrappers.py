import wrapt

from atorch.common.log_utils import default_logger as logger

TORCH_SAVE_WRAPPED = False


@wrapt.decorator
def log_torch_save(wrapped, instance, args, kwargs):
    file_name = None
    if args is not None and len(args) > 1:
        file_name = args[1]

    if kwargs is not None:
        file_name = kwargs.get("f", file_name)

    file_name = file_name or "unknown"

    logger.info(f"Start to {wrapped.__name__} with file_name={file_name}")
    result = wrapped(*args, **kwargs)
    logger.info(f"Finish {wrapped.__name__} with file_name={file_name}, result={result}")
    return result


def wrap_torch_save():
    global TORCH_SAVE_WRAPPED
    if not TORCH_SAVE_WRAPPED:
        import torch

        torch.save = log_torch_save(torch.save)
        TORCH_SAVE_WRAPPED = True
