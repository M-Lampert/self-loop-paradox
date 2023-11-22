import numpy as np
import pandas as pd
import scipy as sp

from stats.walk_statistics import (
    compute_statistics,
    get_avg_closed_walks_ratio,
    get_ratio_estimates,
)


def test_get_avg_closed_walks_ratio():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    walk_length = 2

    result = get_avg_closed_walks_ratio(adjacency_matrix, walk_length, True)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)

    expected_result = [
        (3 / 7 + 2 * (2 / 5)) / 3,
        np.sqrt(
            (
                2 * (((3 / 7 + 2 * (2 / 5)) / 3 - (2 / 5)) ** 2)
                + ((3 / 7 + 2 * (2 / 5)) / 3 - (3 / 7)) ** 2
            )
            / 3
        ),
    ]
    assert np.allclose(result, expected_result)


def test_get_avg_closed_walks_ratio_no_self_loops():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    walk_length = 2
    self_loops = False

    result = get_avg_closed_walks_ratio(
        adjacency_matrix, walk_length, self_loops
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(x, float) for x in result)

    expected_result = [
        (2 * (1 / 2) + 1) / 3,
        np.sqrt((2 * ((2 / 3 - 1 / 2) ** 2) + (2 / 3 - 1) ** 2) / 3),
    ]
    assert np.allclose(result, expected_result)


def test_get_ratio_estimates_walk_length_1_with_self_loops():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    result = get_ratio_estimates(adjacency_matrix, 1, True)

    assert isinstance(result, float)

    expected_result = 1 / (adjacency_matrix.sum(-1).mean() + 1)
    assert np.isclose(result, expected_result)


def test_get_ratio_estimates_walk_length_1_without_self_loops():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    result = get_ratio_estimates(adjacency_matrix, 1, False)

    assert isinstance(result, float)
    assert np.isclose(result, 0)


def test_get_ratio_estimates_walk_length_2_with_self_loops():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    result = get_ratio_estimates(adjacency_matrix, 2, True)

    assert isinstance(result, float)

    mean_degree = adjacency_matrix.sum(-1).mean()
    second_raw_moment = np.square(adjacency_matrix.sum(-1)).mean()
    mean_neighbor_degree = second_raw_moment / mean_degree
    expected_result = (mean_degree + 1) / (
        (mean_degree + 1) * (mean_neighbor_degree + 1)
    )
    assert np.isclose(result, expected_result)


def test_get_ratio_estimates_walk_length_2_without_self_loops():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    result = get_ratio_estimates(adjacency_matrix, 2, False)

    # Check that the function returns a float
    assert isinstance(result, float)

    # Check that the returned value is as expected
    mean_degree = adjacency_matrix.sum(-1).mean()
    second_raw_moment = np.square(adjacency_matrix.sum(-1)).mean()
    expected_result = mean_degree / second_raw_moment
    assert np.isclose(result, expected_result)


def test_get_ratio_estimates_walk_length_above_2():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    walk_length = 3

    # Check that the function raises a NotImplementedError
    try:
        get_ratio_estimates(adjacency_matrix, walk_length, True)
    except NotImplementedError:
        assert True
    else:
        assert False


def test_compute_statistics():
    adjacency_matrix = sp.sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    )
    name = "test_dataset"
    df = pd.DataFrame()
    max_k = 4
    max_est_k = 2

    result = compute_statistics(adjacency_matrix, name, df, max_k, max_est_k)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2 * max_k + max_est_k)
    expected_columns = [
        "1_mean",
        "1_std",
        "2_mean",
        "2_std",
        "3_mean",
        "3_std",
        "4_mean",
        "4_std",
        "1_expected",
        "2_expected",
    ]
    assert all(column in result.columns for column in expected_columns)
    assert list(result.index.names) == ["Dataset", "Self-loops"]
