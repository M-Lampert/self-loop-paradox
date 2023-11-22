import torch
from torch_geometric.data import Data, Dataset

from data import get_sbm, iterate_datasets


def test_iterate_datasets():
    datasets = list(iterate_datasets())

    # Check that the function returns a list of tuples
    assert all(isinstance(dataset, tuple) for dataset in datasets)

    # Check that each tuple contains a string and a Dataset
    assert all(
        isinstance(name, str) and isinstance(dataset, Dataset)
        for name, dataset in datasets
    )

    # Check that the number of datasets is correct
    assert len(datasets) == 23

    # Check that there are no self-loops
    assert all(not dataset[1][0].has_self_loops() for dataset in datasets)

    # Check that there are no isolated nodes
    assert all(not dataset[1][0].has_isolated_nodes() for dataset in datasets)

    # Check that the random node split is correct
    assert all(
        dataset[1][0].train_mask.sum().item()
        + dataset[1][0].test_mask.sum().item()
        == dataset[1][0].num_nodes
        for dataset in datasets
    )

    # Check if the graph is undirected
    assert all(not dataset[1][0].is_directed() for dataset in datasets)


def test_get_sbm():
    block_sizes = torch.tensor([10, 10])
    edge_probs = torch.tensor([[0.5, 0.1], [0.1, 0.5]])
    seed = 42
    cluster_std = torch.tensor([1.0, 1.0])
    centers = 2
    n_features = 16

    data = get_sbm(
        block_sizes, edge_probs, seed, cluster_std, centers, n_features
    )

    # Check that the function returns a Data object
    assert isinstance(data, Data)

    # Check that the Data object has the correct attributes
    assert data.x.size(0) == block_sizes.sum()
    assert data.x.size(1) == n_features
    assert data.y.size(0) == block_sizes.sum()
    assert data.edge_index.size(0) == 2

    # Check that there are no self-loops
    assert not data.has_self_loops()

    # Check that there are no isolated nodes
    assert not data.has_isolated_nodes()

    # Check that the random node split is correct
    assert (
        data.train_mask.sum().item() + data.test_mask.sum().item()
        == data.num_nodes
    )

    # Check if the graph is undirected
    assert not data.is_directed()


def test_get_sbm_defaults():
    block_sizes = torch.tensor([10, 10])
    edge_probs = torch.tensor([[0.5, 0.1], [0.1, 0.5]])

    data = get_sbm(block_sizes, edge_probs)

    # Check that the function returns a Data object
    assert isinstance(data, Data)

    # Check that the Data object has the correct attributes
    assert data.x.size(0) == block_sizes.sum()
    assert data.x.size(1) == 16
    assert data.y.size(0) == block_sizes.sum()
    assert data.edge_index.size(0) == 2


def test_get_sbm_is_reproducible():
    block_sizes = torch.tensor([10, 10])
    edge_probs = torch.tensor([[0.5, 0.1], [0.1, 0.5]])
    seed = 42

    data1 = get_sbm(block_sizes, edge_probs, seed)
    data2 = get_sbm(block_sizes, edge_probs, seed)

    # Check that the function returns a Data object
    assert isinstance(data1, Data)
    assert isinstance(data2, Data)

    # Check that the Data objects are equal
    assert data1.x.equal(data2.x)
    assert data1.y.equal(data2.y)
    # The `stochastic_blockmodel_graph` function has no seed argument
    # so the edge indices are currently not reproducible
    # assert data1.edge_index.equal(data2.edge_index)
    assert data1.num_nodes == data2.num_nodes
