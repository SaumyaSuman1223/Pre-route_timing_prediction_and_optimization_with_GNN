"""
Dataset statistics for the CircuitNet-N14 timing subset (Phase A).

Prints a markdown summary suitable for pasting into the report or a status email:
inventory, per-design scale, the PPA sweep dimensions, target statistics, and the
pre-route-vs-post-route baseline numbers.

    python -m src.dataset_stats --configs 4 > docs/dataset_summary_generated.md
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import circuitnet as C
from .visualize_dataset import collect, to_log

CONFIG_RE = re.compile(
    r"^(?P<design>.+?)_freq_(?P<freq>\d+)_mp_(?P<mp>\d+)_fpu_(?P<fpu>\d+)"
    r"_fpa_(?P<fpa>[\d.]+)_p_(?P<p>\d+)_fi_(?P<fi>\w+)$"
)

SWEEP_MEANING = {
    "freq": "target clock frequency (MHz)",
    "mp": "macro placement variant",
    "fpu": "floorplan utilisation (%)",
    "fpa": "floorplan aspect ratio",
    "p": "placement effort / variant",
    "fi": "filler insertion stage",
}


def sweep_table(configs: List[str]) -> Dict[str, List[str]]:
    vals = defaultdict(set)
    for c in configs:
        m = CONFIG_RE.match(c)
        if m:
            for k in SWEEP_MEANING:
                vals[k].add(m.group(k))
    return {k: sorted(v, key=lambda s: (len(s), s)) for k, v in vals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root", type=Path, default=Path("data/external/circuitnet/timing_extracted")
    )
    p.add_argument("--designs", nargs="+", default=["Vortex-small", "nvdla-small"])
    p.add_argument("--configs", type=int, default=4, help="configs sampled per design")
    args = p.parse_args()

    inventory, runs, graphs = {}, [], {}
    for design in args.designs:
        ddir = args.root / design
        available = C.list_configs(ddir)
        inventory[design] = available
        chosen = available[:: max(1, len(available) // args.configs)][: args.configs]
        for cfg in chosen:
            runs.append(collect(ddir, cfg))
        g = C.load_circuitnet(ddir, chosen[0])
        deg = np.bincount(g.edge_index[0].numpy(), minlength=g.num_nodes)
        graphs[design] = {
            "nodes": g.num_nodes,
            "edges": int(g.edge_index.shape[1]),
            "avg_deg": float(deg.mean()),
            "iso": float((deg == 0).mean() * 100),
            "dropped": int(g.n_dropped_nets),
            "n_feat": g.x.shape[1],
            "n_edge_feat": g.edge_attr.shape[1],
        }

    out: List[str] = []
    A = out.append

    A("## 1. Inventory\n")
    A("| design | configs | sampled here |")
    A("|---|---|---|")
    for d, cfgs in inventory.items():
        n = len([r for r in runs if r["design"] == d])
        A(f"| {d} | {len(cfgs)} | {n} |")
    A(f"| **total** | **{sum(len(v) for v in inventory.values())}** | **{len(runs)}** |\n")

    A("## 2. PPA sweep (what varies within a design)\n")
    sw = sweep_table(sum(inventory.values(), []))
    A("| parameter | meaning | values |")
    A("|---|---|---|")
    for k, meaning in SWEEP_MEANING.items():
        A(f"| `{k}` | {meaning} | {', '.join(sw.get(k, []))} |")
    A("")

    A("## 3. Scale per design (one representative config)\n")
    A("| design | pins | nets (graph nodes) | driver→sink connections | die area (µm²) |")
    A("|---|---|---|---|---|")
    for d in inventory:
        r = next(x for x in runs if x["design"] == d)
        xy = r["driver_xy"]
        ok = (xy[:, 0] > 0) | (xy[:, 1] > 0)
        area = (xy[ok, 0].max() - xy[ok, 0].min()) * (xy[ok, 1].max() - xy[ok, 1].min())
        A(
            f"| {d} | {r['n_pins_place']:,} | {r['n_nets_place']:,} | "
            f"{r['n_edges_place']:,} | {area:,.0f} |"
        )
    A("")

    A("## 4. Built graph (Level-1, after loading)\n")
    A("| design | nodes | edges | avg degree | isolated | dropped nets | node feats | edge feats |")
    A("|---|---|---|---|---|---|---|---|")
    for d, g in graphs.items():
        A(
            f"| {d} | {g['nodes']:,} | {g['edges']:,} | {g['avg_deg']:.2f} | "
            f"{g['iso']:.1f}% | {g['dropped']:,} | {g['n_feat']} | {g['n_edge_feat']} |"
        )
    A("")

    A("## 5. Target: post-route net delay\n")
    raw = np.concatenate([r["route_delay"] for r in runs])
    A(f"- Channels: **{raw.shape[1]}** per net")
    A(f"- Nets measured: **{len(raw):,}** across {len(runs)} configs")
    A(f"- Range: **{raw.min():.6f} to {raw.max():.4f}** (raw units)")
    A(f"- Median **{np.median(raw):.5f}**, mean **{raw.mean():.5f}**, p99 **{np.percentile(raw, 99):.4f}**")
    A(f"- Exactly zero: **{(raw == 0).mean() * 100:.1f}%** of net-channel values")
    A(f"- Distinct values (channel 0): **{len(np.unique(raw[:, 0])):,}** — the timing engine")
    A("  reports on a discrete grid, so the distribution is comb-shaped")
    cm = np.corrcoef(raw.T)
    A(f"- Inter-channel correlation: min **{cm[np.triu_indices(4, 1)].min():.4f}**, "
      f"max **{cm[np.triu_indices(4, 1)].max():.4f}** — the four channels are near-duplicates")
    A("")

    A("## 6. Fanout\n")
    fan = np.concatenate([r["fanout"] for r in runs])
    A(f"- Median **{np.median(fan):.0f}**, mean **{fan.mean():.2f}**, "
      f"p99 **{np.percentile(fan, 99):.0f}**, max **{fan.max():.0f}**")
    A(f"- Fanout-1 nets: **{(fan == 1).mean() * 100:.1f}%**")
    A("")

    A("## 7. The learning problem — pre-route as a predictor of post-route\n")
    A("| channel | MAE (log) | Pearson r | R² of y=x |")
    A("|---|---|---|---|")
    for c in range(4):
        pre = to_log(np.concatenate([r["place_delay"][:, c] for r in runs]))
        post = to_log(np.concatenate([r["route_delay"][:, c] for r in runs]))
        res = post - pre
        A(
            f"| {c} | {np.abs(res).mean():.3f} | {np.corrcoef(pre, post)[0, 1]:.3f} | "
            f"{1 - np.sum(res**2) / np.sum((post - post.mean())**2):.3f} |"
        )
    A("\nThese are the numbers any model must beat: using the pre-route estimate unchanged.\n")

    A("## 8. Stage mismatch\n")
    for d in inventory:
        r = next(x for x in runs if x["design"] == d)
        A(
            f"- **{d}**: pins {r['n_pins_place']:,} → {r['n_pins_route']:,} "
            f"(+{(r['n_pins_route']/r['n_pins_place']-1)*100:.1f}%), "
            f"nets {r['n_nets_place']:,} → {r['n_nets_route']:,} "
            f"(+{(r['n_nets_route']/r['n_nets_place']-1)*100:.1f}%), "
            f"matched by name {r['n_shared']/r['n_nets_place']*100:.1f}%"
        )
    print("\n".join(out))


if __name__ == "__main__":
    main()
