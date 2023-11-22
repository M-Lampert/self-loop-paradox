from pathlib import Path

from utils import get_data_path, get_project_path, get_results_path


def test_get_project_path():
    project_path = get_project_path()
    assert isinstance(project_path, Path)
    # Check if project_path is the correct path
    root_path = Path(".").resolve()
    assert project_path == root_path


def test_get_data_path():
    data_path = get_data_path()
    assert isinstance(data_path, Path)
    assert data_path.exists()


def test_get_results_path():
    results_path = get_results_path()
    assert isinstance(results_path, Path)
    assert results_path.exists()
