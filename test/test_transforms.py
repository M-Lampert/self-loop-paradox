import torch
from torch_geometric.data import Data, HeteroData

from data import RemoveSelfLoops


def test_remove_self_loops():
    assert str(RemoveSelfLoops()) == "RemoveSelfLoops()"

    assert len(RemoveSelfLoops()(Data())) == 0

    edge_index = torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])
    edge_weight = torch.tensor([1, 2, 3, 4, 5, 5, 5])
    edge_attr = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [8, 10], [5, 6]]
    )

    data = Data(edge_index=edge_index, num_nodes=3)
    data = RemoveSelfLoops()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data.num_nodes == 3

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = RemoveSelfLoops()(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data.num_nodes == 3
    assert data.edge_weight.tolist() == [1, 2, 3, 4]

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = RemoveSelfLoops(attr="edge_attr")(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data.num_nodes == 3
    assert data.edge_attr.tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_remove_self_loops_with_only_existing_self_loops():
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=3)
    data = RemoveSelfLoops()(data)
    assert data.edge_index.tolist() == [[], []]
    assert data.num_nodes == 3


def test_remove_self_loops_without_existing_self_loops():
    edge_index = torch.tensor([[2, 0, 1], [0, 1, 2]])
    data = Data(edge_index=edge_index, num_nodes=3)
    data = RemoveSelfLoops()(data)
    assert data.edge_index.tolist() == [[2, 0, 1], [0, 1, 2]]
    assert data.num_nodes == 3


def test_hetero_remove_self_loops():
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]])

    data = HeteroData()
    data["v"].num_nodes = 3
    data["w"].num_nodes = 3
    data["v", "v"].edge_index = edge_index
    data["v", "w"].edge_index = edge_index
    data = RemoveSelfLoops()(data)
    assert data["v", "v"].edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data["v", "w"].edge_index.tolist() == edge_index.tolist()
