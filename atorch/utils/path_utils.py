from pathlib import Path


def path_is_empty(dir_path: Path) -> bool:
    """
    Check whether a Path is empty.
    Args:
        dir_path: a Path to check
    Returns:
        True if dir_path doesn't exist as a dir (could be dir_path is not a dir, no such Path or is None),
            of dir_path exists and has no file inside.
        False if dir_path exists as a dir and is not empty.
    """
    if dir_path is None:
        return True

    if dir_path.exists() and dir_path.is_dir() and next(dir_path.iterdir(), None) is not None:
        return False
    else:
        return True
