# This is mostly copied from the transform `AddSelfLoops` in PyG.
# Only the utility function `remove_self_loops` is used here.
from typing import Optional, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import remove_self_loops


@functional_transform("remove_self_loops")
class RemoveSelfLoops(BaseTransform):
    r"""Removes self-loops from the given homogeneous or heterogeneous graph
    (functional name: :obj:`remove_self_loops`).

    Args:
        attr (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.remove_self_loops`.
            (default: :obj:`"edge_weight"`)
    """

    def __init__(self, attr: Optional[str] = "edge_weight"):
        self.attr = attr

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or "edge_index" not in store:
                continue

            store.edge_index, edge_weight = remove_self_loops(
                store.edge_index, getattr(store, self.attr, None)
            )

            setattr(store, self.attr, edge_weight)

        return data
