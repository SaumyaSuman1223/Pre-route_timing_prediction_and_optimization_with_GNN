"""
Split one large net graph into subgraphs that fit in GPU memory.

Decision D4 called for METIS, but no METIS backend is installed (torch-sparse, pyg-lib
and pymetis are all absent) and building one is a heavy dependency for a laptop 6GB GPU.
SPATIAL tiling is used instead, and it is arguably the better fit: Level-1 edges connect
nets whose bounding boxes OVERLAP, so they are already local in the plane. Cutting the
die into tiles therefore severs very few edges -- the retention rate is measured and
reported by `partition_report` rather than assumed.

Tiles are cut on quantiles, not on a uniform grid, so each partition holds a similar
number of nets even though cells are packed unevenly across a floorplan.
"""

import math
from typing import List

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _quantile_bins(v: Tensor, k: int) -> Tensor:
    """Assign each value to one of k bins with roughly equal population."""
    if k <= 1:
        return torch.zeros(v.numel(), dtype=torch.long)
    qs = torch.quantile(v.float(), torch.linspace(0, 1, k + 1)[1:-1])
    return torch.bucketize(v.contiguous(), qs)


def spatial_partition(data: Data, n_parts: int, x_col: int, y_col: int) -> List[Data]:
    """
    Tile `data` into ~n_parts spatially coherent subgraphs.

    x_col / y_col index the node-position columns of `data.x`; pass them from
    `data.feature_names` rather than hardcoding, since the schema widens in Phase B.
    """
    if n_parts <= 1:
        return [data]

    k = max(1, int(math.sqrt(n_parts)))
    col_bin = _quantile_bins(data.x[:, x_col], k)

    parts: List[Data] = []
    for c in range(k):
        col_mask = col_bin == c
        if not col_mask.any():
            continue
        idx = col_mask.nonzero(as_tuple=True)[0]
        row_bin = _quantile_bins(data.x[idx, y_col], k)
        for r in range(k):
            sel = idx[row_bin == r]
            if sel.numel() < 2:
                continue
            ei, ea = subgraph(
                sel,
                data.edge_index,
                data.edge_attr,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )
            parts.append(Data(x=data.x[sel], edge_index=ei, edge_attr=ea, y=data.y[sel]))
    return parts


def partition_report(data: Data, parts: List[Data]) -> dict:
    """How much of the graph survived tiling. Report this; never assume it."""
    kept_nodes = sum(p.num_nodes for p in parts)
    kept_edges = sum(p.edge_index.shape[1] for p in parts)
    return {
        "n_parts": len(parts),
        "nodes": kept_nodes,
        "node_retention": kept_nodes / max(data.num_nodes, 1),
        "edge_retention": kept_edges / max(data.edge_index.shape[1], 1),
        "max_part_nodes": max((p.num_nodes for p in parts), default=0),
        "max_part_edges": max((p.edge_index.shape[1] for p in parts), default=0),
    }
