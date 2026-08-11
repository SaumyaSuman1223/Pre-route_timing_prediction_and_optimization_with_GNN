"""
CircuitNet-N14 -> Level-1 net graph, in the Phase-A schema.

Everything CircuitNet-specific lives in this file. `dataset.py` re-exports
`load_circuitnet` as the public loader; `src/models/` must never import this module or
know which data source it is training on.

Format notes (all verified empirically -- see docs/phase_a_feature_spec.md section 1):
  nodes.npz['nodes']            (N,) pin-name strings, backslash-escaped. No features.
  net_edges.npz['net_edges']    (E, 6) = [driver_idx, sink_idx, delay x4]
  pin_positions.npz             dict name -> [x1, y1, x2, y2, 0, 0, 0, 0]

Two traps this module exists to handle:
  1. Node index spaces DIFFER between the place and route stages (place 420,652 nodes vs
     route 424,931 for Vortex-small; CTS inserts buffers). Labels are therefore joined by
     PIN NAME, never by index. An index join silently mislabels most of the graph.
  2. ~9% of nets have no counterpart in the other stage. Per decision D3 those are
     dropped, and the count is recorded on the returned object as `n_dropped_nets`.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from . import feature_spec as fs

STAGE_INPUT = "place"
STAGE_LABEL = "route"


# --------------------------------------------------------------------------------------
# raw file access
# --------------------------------------------------------------------------------------


def list_configs(design_dir: Path) -> list:
    """
    Configs that are actually usable: the intersection over every file a graph needs.

    CircuitNet's own tarballs are incomplete and the shortfall differs per feature --
    nvdla-small ships 96 place / 94 route / 78 pin_positions, so 18 configs cannot be
    built at all. Always enumerate through this, never by listing one directory.
    """
    design_dir = Path(design_dir)
    sets = [
        {p.stem for p in (design_dir / stage / feat).glob("*.npz")}
        for stage in (STAGE_INPUT, STAGE_LABEL)
        for feat in ("nodes", "net_edges")
    ]
    sets.append({p.stem for p in (design_dir / "pin_positions").glob("*.npz")})
    return sorted(set.intersection(*sets))


def _load_stage(design_dir: Path, stage: str, config: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pin_names, net_edges) for one stage of one config."""
    nodes = np.load(design_dir / stage / "nodes" / f"{config}.npz", allow_pickle=True)["nodes"]
    edges = np.load(design_dir / stage / "net_edges" / f"{config}.npz")["net_edges"]
    return nodes, edges


def _pin_centers(design_dir: Path, config: str, pin_names: np.ndarray) -> np.ndarray:
    """
    (N, 4) array of [center_x, center_y, bbox_w, bbox_h] for each node, in node-index
    order. pin_positions is keyed by the UNESCAPED name, and only the first 4 of its 8
    columns are ever populated (cols 4-7 are zero across the whole file).
    """
    raw = np.load(
        design_dir / "pin_positions" / f"{config}.npz", allow_pickle=True
    )["pin_positions"].item()

    out = np.zeros((len(pin_names), 4), dtype=np.float32)
    missing = 0
    for i, name in enumerate(pin_names):
        pos = raw.get(name.replace("\\", ""))
        if pos is None:
            missing += 1
            continue
        x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
        out[i] = ((x1 + x2) * 0.5, (y1 + y2) * 0.5, abs(x2 - x1), abs(y2 - y1))
    if missing:
        print(f"  warning: {missing}/{len(pin_names)} pins had no position entry")
    return out


# --------------------------------------------------------------------------------------
# nets
# --------------------------------------------------------------------------------------


def _build_nets(edges: np.ndarray, centers: np.ndarray, n_pins: int):
    """
    A net == its driver pin. The graph is strictly bipartite driver->sink (verified: zero
    nodes have both in- and out-degree, every sink has in-degree exactly 1), so grouping
    edges by column 0 recovers the nets exactly, with no extra file needed.

    Returns (driver_idx, bbox, fanout) where bbox is (n_nets, 4) = x1, y1, x2, y2 over
    the driver pin plus all sink pins.
    """
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)

    order = np.argsort(src, kind="stable")
    src_sorted, dst_sorted = src[order], dst[order]
    driver_idx, starts = np.unique(src_sorted, return_index=True)
    fanout = np.diff(np.append(starts, len(src_sorted))).astype(np.float32)

    # bbox over driver + sinks, accumulated per net with min/max scatter
    n_nets = len(driver_idx)
    net_of_edge = np.repeat(np.arange(n_nets), fanout.astype(np.int64))

    sink_xy = centers[dst_sorted, :2]
    bbox = np.empty((n_nets, 4), dtype=np.float32)
    for col, (fn, init) in enumerate(
        [(np.minimum, np.inf), (np.minimum, np.inf), (np.maximum, -np.inf), (np.maximum, -np.inf)]
    ):
        acc = np.full(n_nets, init, dtype=np.float32)
        fn.at(acc, net_of_edge, sink_xy[:, col % 2])
        bbox[:, col] = acc

    drv_xy = centers[driver_idx, :2]
    bbox[:, 0] = np.minimum(bbox[:, 0], drv_xy[:, 0])
    bbox[:, 1] = np.minimum(bbox[:, 1], drv_xy[:, 1])
    bbox[:, 2] = np.maximum(bbox[:, 2], drv_xy[:, 0])
    bbox[:, 3] = np.maximum(bbox[:, 3], drv_xy[:, 1])

    return driver_idx, bbox, fanout


def _net_labels(edges: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Per-net delay label: the worst (max) delay over the net's driver->sink edges, per
    channel. Net delay is natively per driver-sink pair, but Level-1 nodes are nets, so
    an aggregation is required; max is the timing-meaningful one. Keyed by driver index.
    """
    src = edges[:, 0].astype(np.int64)
    delay = edges[:, 2:6].astype(np.float32)
    drivers, inverse = np.unique(src, return_inverse=True)
    agg = np.zeros((len(drivers), fs.NET_DELAY_CHANNELS), dtype=np.float32)
    np.maximum.at(agg, inverse, delay)
    return {int(d): agg[i] for i, d in enumerate(drivers)}


# --------------------------------------------------------------------------------------
# net-to-net edges (paper section 3.3.1: net_j -> net_i when bounding boxes overlap)
# --------------------------------------------------------------------------------------


def _overlap_edges(
    bbox: np.ndarray, max_degree: int = 16, nets_per_cell: int = 8, seed: int = 0
):
    """
    All-pairs bbox overlap over ~117k nets is 1.4e10 comparisons, so candidates are found
    with a uniform grid and each net keeps at most `max_degree` neighbours (the same
    degree cap CircuitNet's own build_graph.py uses, for the same reason).

    `nets_per_cell` controls candidate PRECISION, and smaller is counter-intuitively
    better: nets are binned by bbox center, so a large cell hands the sampler many
    far-apart nets that then fail the overlap test and yield nothing. Measured on
    Vortex-small (avg degree / isolated nodes): 2 -> 3.75 / 10.7%, 8 -> 8.31 / 3.5%,
    16 -> 8.98 / 2.8%, all at max_degree=16. The defaults sit at the knee.
    """
    rng = np.random.default_rng(seed)
    n = len(bbox)
    cx = (bbox[:, 0] + bbox[:, 2]) * 0.5
    cy = (bbox[:, 1] + bbox[:, 3]) * 0.5

    span_x = max(bbox[:, 2].max() - bbox[:, 0].min(), 1e-6)
    span_y = max(bbox[:, 3].max() - bbox[:, 1].min(), 1e-6)
    cell = float(np.sqrt(span_x * span_y * nets_per_cell / max(n, 1)))

    gx = np.floor((cx - bbox[:, 0].min()) / cell).astype(np.int64)
    gy = np.floor((cy - bbox[:, 1].min()) / cell).astype(np.int64)
    key = gx * (int(span_y / cell) + 2) + gy

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    _, starts = np.unique(key_sorted, return_index=True)
    bounds = np.append(starts, n)

    us, vs = [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        members = order[b0:b1]
        if len(members) < 2:
            continue
        for u in members:
            cand = members[members != u]
            if len(cand) > max_degree:
                cand = rng.choice(cand, max_degree, replace=False)
            # real overlap test, since a shared cell only makes them candidates
            ov = (
                (bbox[cand, 0] <= bbox[u, 2])
                & (bbox[cand, 2] >= bbox[u, 0])
                & (bbox[cand, 1] <= bbox[u, 3])
                & (bbox[cand, 3] >= bbox[u, 1])
            )
            for v in cand[ov]:
                us.append(u)
                vs.append(v)

    if not us:
        return np.zeros((2, 0), dtype=np.int64)
    ei = np.stack([np.array(us), np.array(vs)])
    # the overlap relation is symmetric; the paper's edges are directed, so emit both
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    return np.unique(ei, axis=1)


def _edge_features(bbox: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """[avg_x_i, avg_y_i, avg_x_j, avg_y_j, rel_quadrant] -- progress.md section 4."""
    cx = (bbox[:, 0] + bbox[:, 2]) * 0.5
    cy = (bbox[:, 1] + bbox[:, 3]) * 0.5
    i, j = edge_index[0], edge_index[1]
    dx, dy = cx[j] - cx[i], cy[j] - cy[i]
    quadrant = ((dx >= 0).astype(np.float32) * 2 + (dy >= 0).astype(np.float32))
    return np.stack([cx[i], cy[i], cx[j], cy[j], quadrant], axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------
# public loader
# --------------------------------------------------------------------------------------


def load_circuitnet(
    design_dir: Path,
    config: str,
    max_degree: int = 16,
    nets_per_cell: int = 8,
    cache_dir: Optional[Path] = None,
) -> Data:
    """
    Build the Level-1 net graph for one (design, config) pair.

    design_dir: .../timing_extracted/<design>
    config:     e.g. 'Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap'
    """
    design_dir = Path(design_dir)
    if cache_dir is not None:
        cached = Path(cache_dir) / f"{config}.pt"
        if cached.exists():
            return torch.load(cached, weights_only=False)

    in_names, in_edges = _load_stage(design_dir, STAGE_INPUT, config)
    lb_names, lb_edges = _load_stage(design_dir, STAGE_LABEL, config)

    centers = _pin_centers(design_dir, config, in_names)
    driver_idx, bbox, fanout = _build_nets(in_edges, centers, len(in_names))

    # --- label join BY NAME (trap 1) ---
    label_by_driver_idx = _net_labels(lb_edges)
    label_by_name = {
        lb_names[idx]: vec for idx, vec in label_by_driver_idx.items() if idx < len(lb_names)
    }
    input_driver_names = in_names[driver_idx]
    keep = np.array([nm in label_by_name for nm in input_driver_names])
    n_dropped = int((~keep).sum())

    driver_idx, bbox, fanout = driver_idx[keep], bbox[keep], fanout[keep]
    y_raw = np.stack([label_by_name[nm] for nm in input_driver_names[keep]])

    # the pre-route estimate of the target, keyed the same way (paper Table 1: the
    # pre-route value of the predicted quantity is the first node feature)
    pre_by_driver_idx = _net_labels(in_edges)
    pre_raw = np.stack([pre_by_driver_idx[int(i)] for i in driver_idx])

    # --- node features, Phase-A active columns only ---
    drv = centers[driver_idx]
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    pre_log = np.log(fs.LOG_OFFSET + pre_raw) + fs.LOG_SHIFT  # same space as the target
    x = np.concatenate(
        [
            pre_log,
            np.stack([drv[:, 0], drv[:, 1], w, h, fanout, np.log1p(w + h)], axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    names = fs.active_names(fs.LEVEL1_NODE, phase="A")
    assert x.shape[1] == len(names), f"{x.shape[1]} columns vs {len(names)} declared"

    edge_index = _overlap_edges(bbox, max_degree=max_degree, nets_per_cell=nets_per_cell)
    edge_attr = _edge_features(bbox, edge_index)

    # log-space target, matching CircuitNet's data_graph.py so numbers stay comparable
    y = np.log(fs.LOG_OFFSET + y_raw) + fs.LOG_SHIFT

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y.astype(np.float32)),
    )
    data.feature_names = names
    data.edge_feature_names = fs.active_names(fs.LEVEL1_EDGE, phase="A")
    data.driver_pin_names = input_driver_names[keep].tolist()  # join key for Level 2
    data.config = config
    data.design = design_dir.name
    data.n_dropped_nets = n_dropped
    data.phase = "A"

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(data, Path(cache_dir) / f"{config}.pt")
    return data
