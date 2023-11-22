import numpy as np
import pandas as pd
import scipy as sp


def get_avg_closed_walks_ratio(
    A: sp.sparse.spmatrix, walk_length: int, self_loops: bool
) -> list[float]:
    """Computes the mean and the standard deviation of the ratio of the number
    of closed walks of node `v` of length `walk_length`
    and the total number of walks ending in `v` of length `walk_length`.

    Args:
        A: A numpy array representing the adjacency matrix of a graph.
            The adjacency matrix is expected to not contain self-loops
            and be undirected.
        walk_length: The length of the walks.
        self_loops: Whether to add self loops to the graph.

    Returns:
        The mean and the standard deviation of the ratio of the number of
        closed walks of node `v` of length `walk_length`
        and the total number of walks ending in `v` of length `walk_length`.
    """
    if self_loops:
        A = A + sp.sparse.eye(A.shape[0])

    A_k = A**walk_length
    normalized_A_k = sp.sparse.diags((1 / A_k.sum(-1)).A.squeeze()) @ A_k

    diagonal_mean = normalized_A_k.diagonal().mean()
    diagonal_std = normalized_A_k.diagonal().std()

    return [diagonal_mean, diagonal_std]


def get_ratio_estimates(
    A: sp.sparse.spmatrix, walk_length: int, self_loops: bool
) -> float:
    """Computes the ratio of the expected number of closed walks of length
    `walk_length` and the expected number of total walks ending in the same
    node based on the estimates given in the paper.

    Args:
        A: The adjacency matrix of the graph. The graph is assumed to be
            undirected.
        walk_length: The length of the walks.
        self_loops: Whether the graph contains self-loops or not.

    Raises:
        NotImplementedError: Only implemented for `walk_length` 1 and 2.

    Returns:
        The estimate of the ratio of the expected number of closed walks of
        length `walk_length` and the expected number of total walks ending
        in the same node.
    """
    mean_degree = A.sum(-1).mean()

    if walk_length == 1:
        if self_loops:
            expected_self_fraction = 1 / (mean_degree + 1)
        else:
            expected_self_fraction = 0.0

    elif walk_length == 2:
        second_raw_moment = (A.sum(-1).A ** 2).mean()
        if self_loops:
            mean_neighbor_degree = second_raw_moment / mean_degree
            expected_self_fraction = (1 + mean_degree) / (
                1
                + mean_degree
                + mean_neighbor_degree
                + (mean_neighbor_degree * mean_degree)
            )
        else:
            expected_self_fraction = mean_degree / second_raw_moment

    else:
        raise NotImplementedError()

    return expected_self_fraction


def compute_statistics(
    A: sp.sparse.spmatrix,
    name: str,
    df: pd.DataFrame,
    max_k: int = 6,
    max_est_k: int = 2,
) -> pd.DataFrame:
    """Convenience function that computes the average ratios and their
    estimates for the given graph and saves it in the given dataframe.

    Args:
        A: The adjacency matrix of the graph.
        name: The name of the dataset
        df: The dataframe to save the results in.
        max_k: The maximum length of the walks to consider.
        max_est_k: The maximum length of the walks to consider
        for the estimates.

    Returns:
        The dataframe with the results added.
    """

    sl_list = np.array(
        [get_avg_closed_walks_ratio(A, i, True) for i in range(1, max_k + 1)]
    )
    no_sl_list = np.array(
        [get_avg_closed_walks_ratio(A, i, False) for i in range(1, max_k + 1)]
    )
    sl_expected_list = [
        get_ratio_estimates(A, i, True) for i in range(1, max_est_k + 1)
    ]
    no_sl_expected_list = [
        get_ratio_estimates(A, i, False) for i in range(1, max_est_k + 1)
    ]

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                columns=["Dataset", "Self-loops"]
                + [str(i) + "_mean" for i in range(1, max_k + 1)]
                + [str(i) + "_std" for i in range(1, max_k + 1)]
                + [str(i) + "_expected" for i in range(1, max_est_k + 1)],
                data=[
                    [name, True]
                    + list(sl_list[:, 0])
                    + list(sl_list[:, 1])
                    + sl_expected_list,
                    [name, False]
                    + list(no_sl_list[:, 0])
                    + list(no_sl_list[:, 1])
                    + no_sl_expected_list,
                ],
            ).set_index(["Dataset", "Self-loops"]),
        ]
    )

    return df
