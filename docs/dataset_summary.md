# Dataset summary — CircuitNet-N14 timing subset

Written for reporting the Phase-A data to the supervisor. Numbers are generated, not
hand-copied: `python -m src.dataset_stats --configs 4`. Figures: `results/figures/`
(`python -m src.visualize_dataset`). Schema and decisions: `phase_a_feature_spec.md`.

---

## What this dataset is

CircuitNet-N14 is an **open, industrial-style EDA dataset** released by the CircuitNet
group, built by running real RISC-V/accelerator designs through a commercial
place-and-route flow on a **14nm** technology. We use its **timing feature** subset.

The key property that makes it right for this project: features are dumped at **three
points in the physical-design flow** — after placement, after clock-tree synthesis, and
after routing. "After placement" is exactly what the Chang & Kim paper calls *pre-route*,
and "after routing" is the ground truth it tries to predict. So a single design gives us
an aligned (input, label) pair without running any EDA tool ourselves.

**Why we're on it at all:** the project's real target is our own Cadence Genus/Innovus
data, which does not exist yet. CircuitNet lets the entire model architecture be built
and debugged now, against data of the same *shape*, and swapped later. (Phase A / Phase B
in `progress.md`.)

**What one data point is:** one *net* in one *placed design*. The graph node is a net, its
features are placement geometry and connectivity, and its label is that net's delay after
routing. One design at one PPA setting produces one graph of ~117k–282k nodes.

## Structure

Each design ships **96 configurations** — the same RTL pushed through the flow at
different clock targets, floorplan utilisations, macro placements and placement efforts.
This is what gives us many samples from few designs, and it is also the main limitation:
96 configs of one design are far less diverse than 96 different designs.

Per config, per stage, three arrays:

| File | Contents |
|---|---|
| `nodes.npz` | pin names only — no features, no labels |
| `net_edges.npz` | `(E, 6)`: driver pin index, sink pin index, and **4 net-delay channels** |
| `pin_positions.npz` | pin name → bounding box `[x1, y1, x2, y2]` |

The connection graph is strictly **driver → sink**, and every sink belongs to exactly one
net, so a net is identified by its driver pin and fanout is just the driver's out-degree.

## Headline statistics

| | Vortex-small | nvdla-small |
|---|---|---|
| Usable configs | 96 | 77 |
| Pins | 420,652 | 1,030,662 |
| Nets (= graph nodes) | 117,541 | 282,489 |
| Driver→sink connections | 302,895 | 748,092 |
| Die area | 486,665 µm² | 1,571,812 µm² |
| Net-graph edges (built) | 971,468 | 2,432,404 |
| Avg node degree | 8.3 | 8.6 |

**173 usable graphs total**, ~2.4 M nodes and ~20 M net-graph edges if all are built.
On disk: 3.7 GB (Vortex, all 3 stages) + 4.6 GB (nvdla, place/route only).

**Target — post-route net delay:** 4 channels per net, range 0 → 0.545, median 0.00031,
p99 0.063. 8.4% of values are exactly zero, and only ~2,074 distinct values occur, because
the timing engine reports on a discrete grid — the distribution is comb-shaped, not
smooth. The four channels are near-duplicates (pairwise correlation 0.949–1.000). Trained
in log space, `log(1e-4 + delay) + 9.211`, matching the reference implementation.

**Fanout:** median 1, mean 2.59, p99 32, max 13,652. 66.8% of nets are point-to-point.

## The learning problem, quantified

The honest baseline is "just use the pre-route estimate and skip the model." Across 8
configs and 1.6 M nets:

| channel | MAE (log space) | Pearson r | R² of y = x |
|---|---|---|---|
| 0 | 0.312 | 0.925 | 0.846 |
| 1 | 0.313 | 0.926 | 0.847 |
| 2 | 0.314 | 0.923 | 0.839 |
| 3 | 0.315 | 0.924 | 0.841 |

So routing leaves the pre-route estimate **already fairly good (r ≈ 0.92)** but not good
enough — and the residual is what the GNN has to model. Any result we report has to be
compared against this line, not against zero. `results/figures/fig2_preroute_vs_postroute.png`
is this table as a picture.

## Honest limitations — state these to the supervisor

1. **No net R and no net C.** The paper's Level-1 predicts resistance and capacitance.
   CircuitNet ships neither, nor RUDY or via counts. Phase A therefore predicts **net
   delay** instead — same architecture, same role in the two-stage pipeline, different
   target. The real targets come back in Phase B from Cadence SPEF.
2. **No cell/timing-arc edges.** Only net connections. The Level-2 arc graph has to be
   synthesized from instance names in Phase A; Cadence gives real arcs later.
3. **Two designs, not many.** 173 graphs sounds like a lot, but they are 2 designs × ~87
   configs. Config-level generalization is well covered; design-level generalization is
   testable in only the weakest sense.
4. **The dataset itself is incomplete.** nvdla-small ships 96 placement configs but only
   94 route and 78 pin_position files, so 19 configs cannot be built. Configs are
   enumerated by intersecting all required files (`circuitnet.list_configs`).
5. **Pre- and post-route are not the same netlist.** CTS inserts buffers, so pin counts
   grow 1.0–2.4% and node ordering differs between stages. Labels are joined by pin name;
   an index-based join would silently mislabel most of the graph. 99.8–100% of pre-route
   nets find a post-route match.

## Figures

| File | Shows |
|---|---|
| `fig1_dataset_scale.png` | inventory and scale |
| `fig2_preroute_vs_postroute.png` | **the learning problem** — pre vs post, and the y=x baseline |
| `fig3_label_distributions.png` | target distribution, raw vs log; the quantization comb |
| `fig4_fanout_and_degree.png` | fanout tail and the built graph's connectivity |
| `fig5_spatial_delay.png` | delay across the die — visible macros and structure |
| `fig6_stage_mismatch.png` | why labels are joined by name |
