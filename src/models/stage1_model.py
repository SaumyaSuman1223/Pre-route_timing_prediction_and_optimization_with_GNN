"""
Level 1: per-net regression from pre-route (post-placement) features.

Paper section 3.3 -- three GNN layers over the net-based graph. In the paper this level
predicts net R and net C. CircuitNet ships neither, so in Phase A the target is
post-route NET DELAY (4 channels); the architecture and its role in the two-stage
pipeline are unchanged. See docs/phase_a_feature_spec.md.

The input width is read from the feature spec, never hardcoded, so that Phase B's extra
columns (net_R, net_C, RUDY, via counts) widen the model without an edit here.
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data import feature_spec as fs
from .net_arc_conv import NetArcConv


class Stage1Model(nn.Module):
    """3 x NetArcConv + an MLP head, per paper section 3.3."""

    def __init__(
        self,
        in_dim: int = None,
        edge_dim: int = None,
        out_dim: int = fs.NET_DELAY_CHANNELS,
        hidden: int = 32,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.0,
        phase: fs.Phase = "A",
    ):
        super().__init__()
        # width comes from the declared schema unless explicitly overridden
        in_dim = in_dim if in_dim is not None else fs.active_dim(fs.LEVEL1_NODE, phase)
        edge_dim = edge_dim if edge_dim is not None else fs.active_dim(fs.LEVEL1_EDGE, phase)
        self.in_dim, self.edge_dim, self.out_dim = in_dim, edge_dim, out_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, hidden)
        )
        self.convs = nn.ModuleList(
            [
                NetArcConv(hidden, hidden // heads, edge_dim, heads=heads, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.leaky_relu(norm(conv(h, edge_index, edge_attr)), 0.2)
        return self.head(h)

    @property
    def feature_names(self) -> List[str]:
        return fs.active_names(fs.LEVEL1_NODE, "A")


class SingleStageBaseline(Stage1Model):
    """
    The ablation partner: identical capacity, but it is trained to predict the Level-2
    target directly from pre-route features in one shot, with no intermediate Level-1
    prediction feeding it. Defined here so the two arms cannot drift apart in width or
    depth -- any accuracy gap is then attributable to the two-stage structure and not to
    one model simply being bigger.
    """
