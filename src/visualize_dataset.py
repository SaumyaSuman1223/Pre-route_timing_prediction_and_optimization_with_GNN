"""
Dataset-characterisation figures for CircuitNet-N14 (Phase A).

Produces the figure set that answers "what is this data, and what is the learning
problem?" -- intended for the project report and for showing the supervisor. This
describes the DATASET only; model-result plots live in src/evaluate.py.

    python -m src.visualize_dataset                      # defaults, ~2 min
    python -m src.visualize_dataset --configs 4 --out results/figures

Figures written to results/figures/:
    fig1_dataset_scale.png        inventory and per-design scale
    fig2_preroute_vs_postroute.png  THE plot: what routing does to timing
    fig3_label_distributions.png  target distribution, raw and log-transformed
    fig4_fanout_and_degree.png    net fanout and net-graph connectivity
    fig5_spatial_delay.png        where the slow nets are on the die
    fig6_stage_mismatch.png       why labels must be joined by pin name
"""

import argparse
import textwrap
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter

from .data import circuitnet as C
from .data import feature_spec as fs

# --------------------------------------------------------------------------------------
# palette -- validated with the dataviz skill's validate_palette.js (light mode):
# all checks PASS; two hues sit under 3:1 contrast, so every series is DIRECT-LABELLED,
# which is the required relief. Figures are light-mode only by choice (they are going
# into a printed report and slides, not a themed web page).
# --------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e5e4e0"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

CHANNEL_LABELS = ["rise / early", "fall / early", "rise / late", "fall / late"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # never dashed
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 160,
        }
    )
    for spine in ("top", "right"):
        plt.rcParams[f"axes.spines.{spine}"] = False


def _finish(ax, note: str = "", width: int = 95) -> None:
    """
    Footnote under an axes. Wrapped, because a long single-line annotation extends past
    the figure and bbox_inches="tight" then grows the canvas to fit it, squashing the
    plots it was meant to explain.
    """
    if note:
        wrapped = "\n".join(
            textwrap.fill(para, width) for para in note.split("\n")
        )
        ax.annotate(
            wrapped,
            xy=(0, -0.19),
            xycoords="axes fraction",
            fontsize=8.5,
            color=INK_MUTED,
            va="top",
            linespacing=1.5,
        )


# --------------------------------------------------------------------------------------
# data collection
# --------------------------------------------------------------------------------------


def _per_net_delay(edges: np.ndarray, names: np.ndarray) -> Dict[str, np.ndarray]:
    """Worst-case (max over sinks) net delay per channel, keyed by driver pin NAME."""
    by_idx = C._net_labels(edges)
    return {names[i]: v for i, v in by_idx.items() if i < len(names)}


def collect(design_dir: Path, config: str) -> dict:
    """Everything the figures need for one (design, config), joined by pin name."""
    pl_names, pl_edges = C._load_stage(design_dir, "place", config)
    rt_names, rt_edges = C._load_stage(design_dir, "route", config)
    centers = C._pin_centers(design_dir, config, pl_names)

    place = _per_net_delay(pl_edges, pl_names)
    route = _per_net_delay(rt_edges, rt_names)
    shared = [n for n in place if n in route]

    idx_of = {n: i for i, n in enumerate(pl_names)}
    driver_idx = np.array([idx_of[n] for n in shared])
    src = pl_edges[:, 0].astype(np.int64)
    fanout_by_idx = np.bincount(src, minlength=len(pl_names))

    return {
        "design": design_dir.name,
        "config": config,
        "place_delay": np.stack([place[n] for n in shared]),
        "route_delay": np.stack([route[n] for n in shared]),
        "driver_xy": centers[driver_idx, :2],
        "fanout": fanout_by_idx[driver_idx].astype(np.float64),
        "n_pins_place": len(pl_names),
        "n_pins_route": len(rt_names),
        "n_nets_place": len(place),
        "n_nets_route": len(route),
        "n_shared": len(shared),
        "n_edges_place": len(pl_edges),
        "n_edges_route": len(rt_edges),
    }


def to_log(x: np.ndarray) -> np.ndarray:
    return np.log(fs.LOG_OFFSET + x) + fs.LOG_SHIFT


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def fig1_scale(runs: List[dict], graphs: Dict[str, dict], out: Path) -> None:
    """Inventory. This is a table, not a chart -- the data's job here is lookup."""
    designs = sorted({r["design"] for r in runs})
    rows = []
    for d in designs:
        r = next(x for x in runs if x["design"] == d)
        g = graphs[d]
        rows.append(
            [
                d,
                f"{g['n_configs']}",
                f"{r['n_pins_place']:,}",
                f"{r['n_nets_place']:,}",
                f"{r['n_edges_place']:,}",
                f"{g['n_graph_edges']:,}",
                f"{g['avg_deg']:.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 1.1 + 0.5 * len(rows)))
    ax.axis("off")
    cols = [
        "design",
        "configs",
        "pins",
        "nets\n(= graph nodes)",
        "driver→sink\nconnections",
        "net-graph edges\n(bbox overlap)",
        "avg\ndegree",
    ]
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="right", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.9)
    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(INK if row == 0 else INK_2)
        if row == 0:
            cell.get_text().set_weight("600")
            cell.get_text().set_color(INK)
        if col == 0:
            cell.get_text().set_ha("left")
    ax.set_title(
        "CircuitNet-N14 timing subset — one graph per (design, config)", pad=16
    )
    ax.annotate(
        "Per-config figures shown for one representative config; all configs are the same design "
        "at different clock/floorplan settings.",
        xy=(0, -0.05),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_dataset_scale.png", bbox_inches="tight")
    plt.close(fig)


def fig2_preroute_vs_postroute(runs: List[dict], out: Path) -> None:
    """
    The headline figure: pre-route estimate vs post-route truth, per net.

    Density (one hue, light->dark) rather than a scatter, because ~250k points overplot
    into a solid blob. The y=x line is the "use the pre-route number as-is" baseline --
    the vertical spread around it IS the problem this project is trying to predict.
    """
    pre = to_log(np.concatenate([r["place_delay"][:, 2] for r in runs]))
    post = to_log(np.concatenate([r["route_delay"][:, 2] for r in runs]))

    resid = post - pre
    mae = np.abs(resid).mean()
    r = np.corrcoef(pre, post)[0, 1]
    r2_identity = 1 - np.sum(resid**2) / np.sum((post - post.mean()) ** 2)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    lo, hi = np.percentile(np.concatenate([pre, post]), [0.2, 99.8])
    hb = ax.hexbin(
        pre, post, gridsize=110, extent=(lo, hi, lo, hi), cmap=SEQ, bins="log", mincnt=1
    )
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2, zorder=5)
    ax.annotate(
        "y = x — pre-route estimate used as-is",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_2,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("pre-route (post-placement) net delay  [log-transformed]")
    ax.set_ylabel("post-route net delay  [log-transformed]")
    ax.set_title("Routing changes net delay — this gap is the learning problem")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("nets per bin (log scale)", color=INK_2, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelcolor=INK_MUTED, labelsize=8)

    r_lo, r_hi = np.percentile(resid, [0.1, 99.9])
    ax2.hist(resid, bins=140, range=(r_lo, r_hi), color=SERIES[0], linewidth=0)
    ax2.axvline(0, color=INK_2, linewidth=1.2)
    ax2.set_xlim(r_lo, r_hi)
    ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax2.set_xlabel("post-route − pre-route  [log-transformed]")
    ax2.set_ylabel("nets")
    ax2.set_title("Error if you skip the model")
    ax2.annotate(
        f"MAE {mae:.3f}\nPearson r {r:.3f}\nR² of y=x  {r2_identity:.3f}",
        xy=(0.97, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        color=INK,
        linespacing=1.6,
    )
    _finish(
        ax,
        f"Channel 2 (rise/late), {len(pre):,} nets across {len(runs)} configs. "
        "Pre- and post-route nets joined by driver pin name.",
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_preroute_vs_postroute.png", bbox_inches="tight")
    plt.close(fig)


def fig3_labels(runs: List[dict], out: Path) -> None:
    """Target distribution. Raw is unusable directly; this is why we train in log space."""
    raw = np.concatenate([r["route_delay"] for r in runs])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for c in range(fs.NET_DELAY_CHANNELS):
        ax.hist(
            raw[:, c],
            bins=160,
            histtype="step",
            linewidth=1.6,
            color=SERIES[c],
            label=CHANNEL_LABELS[c],
        )
    ax.set_yscale("log")
    ax.set_xlabel("post-route net delay  [raw]")
    ax.set_ylabel("nets (log scale)")
    ax.set_title("Raw target — mass piled at zero, long tail")

    logged = to_log(raw)
    for c in range(fs.NET_DELAY_CHANNELS):
        counts, edges = np.histogram(logged[:, c], bins=110)
        mids = (edges[:-1] + edges[1:]) / 2
        ax2.plot(mids, counts, color=SERIES[c], linewidth=1.8)
        # direct labels: the required relief for the two low-contrast hues, and it
        # removes the need to map four near-identical curves through a legend box
        peak = mids[counts.argmax()]
        ax2.annotate(
            CHANNEL_LABELS[c],
            xy=(peak, counts.max()),
            xytext=(6, -2 - 13 * c),
            textcoords="offset points",
            fontsize=8.5,
            color=SERIES[c],
            weight="600",
        )
    ax2.set_xlabel("post-route net delay  [log-transformed]")
    ax2.set_ylabel("nets")
    ax2.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax2.set_title("After log transform — trainable")

    ax.legend(title=None, loc="upper right")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _finish(
        ax2,
        "The four channels are near-identical (pairwise r ≥ 0.95) so the curves overlap. The comb "
        "pattern is real: the timing engine reports delay on a discrete grid.",
    )
    fig.tight_layout()
    fig.savefig(out / "fig3_label_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig4_fanout_degree(runs: List[dict], graphs: Dict[str, dict], out: Path) -> None:
    """Two different connectivity stories, so two panels on their own axes."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    fan = np.concatenate([r["fanout"] for r in runs])
    # log-spaced bins: the raw per-value counts in the tail are single-net noise
    bins = np.unique(np.round(np.logspace(0, np.log10(fan.max() + 1), 60)))
    counts, edges = np.histogram(fan, bins=bins)
    ax.plot(edges[:-1], np.maximum(counts, 0.5), color=SERIES[0], linewidth=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("net fanout (driver out-degree)")
    ax.set_ylabel("nets")
    ax.set_title("Net fanout")
    ax.annotate(
        f"median {np.median(fan):.0f}   p99 {np.percentile(fan, 99):.0f}   max {fan.max():.0f}",
        xy=(0.97, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9.5,
        color=INK,
    )

    for i, (design, g) in enumerate(sorted(graphs.items())):
        vals, counts = np.unique(g["degrees"], return_counts=True)
        ax2.plot(vals, counts, color=SERIES[i], linewidth=1.8, label=design)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1)
    ax2.set_xlabel("net-graph degree (overlapping-bbox neighbours)")
    ax2.set_ylabel("nets")
    ax2.set_title("Net-graph connectivity")
    ax2.legend(loc="upper right")
    _finish(
        ax2,
        "Edges connect nets whose bounding boxes overlap (paper §3.3.1), found via a spatial grid. "
        "Sampling is capped at 16 neighbours per net, but the relation is symmetrised afterwards, "
        "so the final degree can exceed the cap.",
    )
    fig.tight_layout()
    fig.savefig(out / "fig4_fanout_and_degree.png", bbox_inches="tight")
    plt.close(fig)


def fig5_spatial(runs: List[dict], out: Path) -> None:
    """Magnitude over space -> sequential one-hue ramp, aggregated into hex bins."""
    designs = sorted({r["design"] for r in runs})
    fig, axes = plt.subplots(1, len(designs), figsize=(6.2 * len(designs), 5.4))
    axes = np.atleast_1d(axes)

    for ax, design in zip(axes, designs):
        sub = [r for r in runs if r["design"] == design]
        xy = np.concatenate([r["driver_xy"] for r in sub])
        d = to_log(np.concatenate([r["route_delay"][:, 2] for r in sub]))
        ok = (xy[:, 0] > 0) | (xy[:, 1] > 0)  # drop the few pins with no position entry
        hb = ax.hexbin(
            xy[ok, 0], xy[ok, 1], C=d[ok], gridsize=64, cmap=SEQ, reduce_C_function=np.mean
        )
        ax.set_aspect("equal")
        ax.set_title(f"{design} — mean post-route net delay by location")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.grid(False)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.86)
        cb.set_label("net delay [log-transformed]", color=INK_2, fontsize=8.5)
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=GRID, labelcolor=INK_MUTED, labelsize=8)

    axes[0].annotate(
        "Delay has spatial structure — which is the premise for using placement geometry as input.",
        xy=(0, -0.14),
        xycoords="axes fraction",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out / "fig5_spatial_delay.png", bbox_inches="tight")
    plt.close(fig)


def fig6_stage_mismatch(runs: List[dict], out: Path) -> None:
    """
    Why labels must be joined by pin name.

    Plotted as PERCENT CHANGE, not raw counts: the two designs differ in size by 2.4x, so
    raw counts on a shared axis compare nothing. One axis, one unit, directly comparable.
    """
    designs = sorted({r["design"] for r in runs})
    metrics = [
        ("pins", "n_pins_place", "n_pins_route"),
        ("nets", "n_nets_place", "n_nets_route"),
        ("driver→sink connections", "n_edges_place", "n_edges_route"),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    height = 0.34
    ys = np.arange(len(metrics), dtype=float)

    for i, design in enumerate(designs):
        r = next(x for x in runs if x["design"] == design)
        pct = [(r[b] - r[a]) / r[a] * 100 for _, a, b in metrics]
        offset = (i - (len(designs) - 1) / 2) * (height + 0.03)
        bars = ax.barh(ys + offset, pct, height, color=SERIES[i], linewidth=0, label=design)
        for bar, v in zip(bars, pct):
            ax.annotate(
                f"+{v:.1f}%",
                xy=(v, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )

    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlabel("change from post-placement to post-route  (%)")
    ax.set_title("Pre-route and post-route are not the same netlist")
    ax.set_xlim(0, max(ax.get_xlim()[1] * 1.12, 1))
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)

    frac = np.mean([r["n_shared"] / r["n_nets_place"] for r in runs]) * 100
    _finish(
        ax,
        "Clock-tree synthesis inserts buffers between the two stages, so the netlists grow and "
        f"node indices do NOT correspond — labels must be joined by pin name.\n{frac:.1f}% of "
        "pre-route nets have a post-route counterpart; the rest are dropped.",
    )
    fig.tight_layout()
    fig.savefig(out / "fig6_stage_mismatch.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/external/circuitnet/timing_extracted"),
    )
    p.add_argument("--designs", nargs="+", default=["Vortex-small", "nvdla-small"])
    p.add_argument("--configs", type=int, default=2, help="configs sampled per design")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)

    runs: List[dict] = []
    graphs: Dict[str, dict] = {}
    for design in args.designs:
        ddir = args.root / design
        available = C.list_configs(ddir)
        chosen = available[:: max(1, len(available) // args.configs)][: args.configs]
        for cfg in chosen:
            print(f"[collect] {design} / {cfg}")
            runs.append(collect(ddir, cfg))

        print(f"[graph]   {design} / {chosen[0]}")
        g = C.load_circuitnet(ddir, chosen[0])
        deg = np.bincount(g.edge_index[0].numpy(), minlength=g.num_nodes)
        graphs[design] = {
            "n_configs": len(available),
            "n_graph_edges": int(g.edge_index.shape[1]),
            "avg_deg": float(g.edge_index.shape[1] / g.num_nodes),
            "degrees": deg,
        }

    print("[render]  figures")
    fig1_scale(runs, graphs, args.out)
    fig2_preroute_vs_postroute(runs, args.out)
    fig3_labels(runs, args.out)
    fig4_fanout_degree(runs, graphs, args.out)
    fig5_spatial(runs, args.out)
    fig6_stage_mismatch(runs, args.out)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
