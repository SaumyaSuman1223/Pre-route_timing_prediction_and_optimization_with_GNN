"""
Custom MessagePassing layer -- the paper's core GNN operator (Chang & Kim, Eq. 2-5).

Used by BOTH levels, which is why it is called net/arc rather than net or arc:
  Level 1 -- nodes are nets,  edges connect nets whose bounding boxes overlap
  Level 2 -- nodes are arcs,  edges connect arcs inside overlapping nets

Both graphs carry the same 5 edge features, so one operator serves both. The layer is
GAT-style attention CONDITIONED ON EDGE FEATURES: the paper's edges carry the relative
geometry of the two nets (their centres and the relative-position quadrant), which is
precisely the signal that should decide how much neighbour j matters to node i. A plain
GAT that ignores edge_attr throws that away.

Eq. (2)-(5), as implemented here:
    (2)  z_i      = W h_i                                    linear projection
    (3)  e_ij     = LeakyReLU( a^T [ z_i || z_j || U f_ij ] ) attention logit
    (4)  alpha_ij = softmax_j( e_ij )                         normalise over neighbours
    (5)  h_i'     = || _k  sum_j alpha^k_ij z^k_j             multi-head aggregation

Deviation worth knowing: the paper does not state its head count or hidden width, so
those are ours (progress.md section 7 -- hyperparameters are not given in the paper and
are chosen empirically).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class NetArcConv(MessagePassing):
    """One attention layer over a net graph or an arc graph."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_node = nn.Linear(in_dim, heads * out_dim, bias=False)  # W, Eq. (2)
        self.lin_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)  # U, Eq. (3)

        # a, Eq. (3) -- split into the three blocks it multiplies so no concat is needed
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_edge = nn.Parameter(torch.empty(1, heads, out_dim))

        width = heads * out_dim if concat else out_dim
        self.bias = nn.Parameter(torch.zeros(width))
        # a residual keeps 3 stacked layers trainable; projected when the width changes
        self.res = nn.Identity() if in_dim == width else nn.Linear(in_dim, width, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.lin_node, self.lin_edge):
            nn.init.xavier_uniform_(lin.weight)
        for att in (self.att_src, self.att_dst, self.att_edge):
            nn.init.xavier_uniform_(att)
        nn.init.zeros_(self.bias)
        if isinstance(self.res, nn.Linear):
            nn.init.xavier_uniform_(self.res.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        n, h, d = x.size(0), self.heads, self.out_dim

        z = self.lin_node(x).view(n, h, d)  # Eq. (2)
        e = self.lin_edge(edge_attr).view(-1, h, d)

        alpha_src = (z * self.att_src).sum(-1)
        alpha_dst = (z * self.att_dst).sum(-1)
        alpha_edge = (e * self.att_edge).sum(-1)

        out = self.propagate(
            edge_index, z=z, alpha=(alpha_src, alpha_dst), alpha_edge=alpha_edge, size=None
        )
        out = out.reshape(n, h * d) if self.concat else out.mean(dim=1)
        return out + self.bias + self.res(x)

    def message(
        self,
        z_j: Tensor,
        alpha_j: Tensor,
        alpha_i: Tensor,
        alpha_edge: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        alpha = F.leaky_relu(alpha_j + alpha_i + alpha_edge, self.negative_slope)  # (3)
        alpha = softmax(alpha, index, ptr, size_i)  # Eq. (4)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return z_j * alpha.unsqueeze(-1)  # Eq. (5); summed by aggr="add"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_dim} -> {self.out_dim} x "
            f"{self.heads} heads)"
        )
