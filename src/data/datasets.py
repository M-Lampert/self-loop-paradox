from typing import Iterator, Optional

import torch
from sklearn.datasets import make_blobs
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import (
    Actor,
    Airports,
    Amazon,
    CitationFull,
    Coauthor,
    HeterophilousGraphDataset,
    LastFMAsia,
    Planetoid,
    Twitch,
    WebKB,
    WikipediaNetwork,
)
from torch_geometric.transforms import (
    Compose,
    RandomNodeSplit,
    RemoveIsolatedNodes,
    ToUndirected,
)
from torch_geometric.utils import stochastic_blockmodel_graph

from data.transforms import RemoveSelfLoops
from utils import get_data_path


def iterate_datasets() -> Iterator[tuple[str, Dataset]]:
    """
    Returns an iterator that iterates over all real-world
    datasets that were used in the empirical evaluation in Appendix C.
    The datasets were chosen based on the following criteria:
    - Available in PyTorch Geometric
    - Consists of a single graph
    - Contains node features
    - Preferably undirected and a multi-class classification
    - Not too small or too big
    Note that the goal was not to get all datasets that satisfy these criteria,
    but to get a diverse set of datasets for the empirical evaluation.
    For some datasets, the reason why they were not included is given below.

    Yields:
        A tuple of the dataset name and the dataset itself.
    """
    data_path = get_data_path()
    dataset_list, dataset_names = [], []
    transforms_list = [
        RemoveIsolatedNodes(),
        RemoveSelfLoops(),
        RandomNodeSplit(split="train_rest", num_val=0, num_test=0.2),
    ]
    transform = Compose(transforms_list)
    transform_directed = Compose([ToUndirected()] + transforms_list)

    dataset_list += [
        Planetoid(root=data_path / "Planetoid", name=name, transform=transform)
        for name in ["Cora", "CiteSeer", "PubMed"]
    ]
    dataset_names += [
        "Planetoid: Cora",
        "Planetoid: CiteSeer",
        "Planetoid: PubMed",
    ]

    dataset_list += [
        CitationFull(
            root=data_path / "CitationFull", name=name, transform=transform
        )
        for name in ["DBLP"]
    ]
    dataset_names += ["CitationFull: DBLP"]

    dataset_list += [
        Amazon(root=data_path / "Amazon", name=name, transform=transform)
        for name in ["Computers", "Photo"]
    ]
    dataset_names += ["Amazon: Computers", "Amazon: Photo"]

    # Coathour: Physics does not fit into CPU memory for statistics computation
    dataset_list += [
        Coauthor(root=data_path / "Coauthor", name=name, transform=transform)
        for name in ["CS"]
    ]
    dataset_names += ["Coauthor: CS"]

    dataset_list += [
        WikipediaNetwork(
            root=data_path / "Wikipedia",
            name=name,
            transform=transform_directed,
        )
        for name in ["chameleon", "squirrel"]
    ]
    dataset_names += ["Wikipedia: Chameleon", "Wikipedia: Squirrel"]

    # Does not fit into CPU memory for statistics computation
    # dataset_list += [Flickr(root=data_path / "Flickr", transform=transform)]
    # dataset_names += ["Flickr"]

    # Invalid number formats
    # dataset_list += [Yelp(root=data_path / "Yelp", transform=transform)]
    # dataset_names += ["Yelp"]

    dataset_list += [
        Actor(root=data_path / "Actor", transform=transform_directed)
    ]
    dataset_names += ["Actor"]

    # "Europe" and "Brazil" somehow produce a division by zero error
    dataset_list += [
        Airports(
            root=data_path / "Airports",
            name=name,
            transform=transform_directed,
        )
        for name in ["USA", "Europe", "Brazil"]
    ]
    dataset_names += ["Airports: USA", "Airports: Europe", "Airports: Brazil"]

    dataset_list += [
        Twitch(root=data_path / "Twitch", name=name, transform=transform)
        for name in ["DE", "EN", "RU"]
    ]
    dataset_names += ["Twitch: DE", "Twitch: EN", "Twitch: RU"]

    # Too big: Out of memory error
    # dataset_list += [Reddit(root=data_path / "Reddit", transform=transform)]
    # dataset_names += ["Reddit"]

    dataset_list += [
        WebKB(
            root=data_path / "WebKB", name=name, transform=transform_directed
        )
        for name in ["Cornell", "Texas", "Wisconsin"]
    ]
    dataset_names += ["WebKB: Cornell", "WebKB: Texas", "WebKB: Wisconsin"]

    # Does not fit into CPU memory for statistics computation
    # dataset_list += [GitHub(root=data_path / "GitHub", transform=transform)]
    # dataset_names += ["GitHub"]

    # Does not fit into CPU memory for statistics computation
    # dataset_list += [
    #     FacebookPagePage(
    #         root=data_path / "FacebookPagePage", transform=transform
    #     )
    # ]
    # dataset_names += ["FacebookPagePage"]

    dataset_list += [
        LastFMAsia(root=data_path / "LastFMAsia", transform=transform)
    ]
    dataset_names += ["LastFMAsia"]

    dataset_list += [
        HeterophilousGraphDataset(
            root=data_path / "HeterophilousGraphDataset",
            name=name,
            transform=transform_directed,
        )
        for name in ["Amazon-ratings", "Roman-empire", "Minesweeper"]
    ]
    dataset_names += ["Amazon-Ratings", "Roman-Empire", "Minesweeper"]

    for name, dataset in zip(dataset_names, dataset_list):
        yield name, dataset


def get_sbm(
    block_sizes: torch.Tensor,
    edge_probs: torch.Tensor,
    seed: int = 42,
    cluster_std: Optional[torch.Tensor] = None,
    centers: Optional[torch.Tensor] = None,
    n_features=16,
) -> Data:
    """Creates a synthetic dataset using the stochastic blockmodel.
    The node features are sampled from Gaussian distributions.

    Args:
        block_sizes: The number of nodes per block.
        edge_probs: The probability of edges between blocks given as matrix.
            The probability of each possible edge between block `i` and
            block `j` is `edge_probs[i, j]`.
        seed: The random seed. Defaults to 42.
        cluster_std: The standard deviation of each Gaussian distribution.
            If None, all distributions have the standard deviation `1`.
            Defaults to None.
        centers: The centers of each Gaussian distribution.
            If None, the centers are chosen randomly.
            Defaults to None.
        n_features: The number of features per node generated
            from the Gaussian distribution. Defaults to 16.

    Returns:
        A PyG Data object containing the synthetic dataset.
    """
    num_samples = int(block_sizes.sum())
    num_classes = block_sizes.size(0)

    if cluster_std is None:
        cluster_std = torch.ones(num_classes)

    if centers is None:
        centers = num_classes

    X, y = make_blobs(
        n_samples=num_samples,
        centers=centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=seed,
        shuffle=False,
    )
    X = torch.from_numpy(X).to(torch.float)
    y = torch.from_numpy(y).to(torch.long)

    edge_index = stochastic_blockmodel_graph(
        block_sizes=block_sizes, edge_probs=edge_probs, directed=False
    )
    data = Data(x=X, y=y, edge_index=edge_index, num_nodes=num_samples)

    data = RandomNodeSplit(split="train_rest", num_val=0, num_test=0.2)(data)
    data = RemoveIsolatedNodes()(data)
    data = RemoveSelfLoops()(data)

    return data
