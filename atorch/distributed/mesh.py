from atorch.common.log_utils import default_logger as logger
from atorch.utils.import_util import is_torch_npu_available

try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    init_device_mesh = None


def build_mesh(slicing_dim, pg_name_prefix="", reverse_mesh_pg_order=True, device_type="cuda"):
    device_type = "npu" if is_torch_npu_available() else device_type

    dims = []
    names = []
    for item in slicing_dim:
        name = pg_name_prefix + item[0]
        d = item[1]
        if d > 1:
            dims.append(d)
            names.append(name)
    if reverse_mesh_pg_order:
        dims.reverse()
        names.reverse()
    logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
    names = tuple(names)
    return init_device_mesh(device_type, dims, mesh_dim_names=names)
