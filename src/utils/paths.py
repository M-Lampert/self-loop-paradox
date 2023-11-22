from pathlib import Path


def get_project_path() -> Path:
    """Get the path to the project root directory.

    Returns:
        The path to the project root directory as `Path` object.
    """
    file_path = Path(__file__).resolve()
    utils_dir_path = file_path.parent
    src_dir_path = utils_dir_path.parent
    project_dir_path = src_dir_path.parent
    return project_dir_path


def get_data_path() -> Path:
    """Get the path to the data directory.

    Returns:
        The path to the data directory as `Path` object.
    """
    data_path = get_project_path() / "data"
    data_path.mkdir(exist_ok=True)
    return data_path


def get_results_path() -> Path:
    """Get the path to the results directory.

    Returns:
        The path to the results directory as `Path` object.
    """
    result_path = get_project_path() / "results"
    result_path.mkdir(exist_ok=True)
    return result_path
