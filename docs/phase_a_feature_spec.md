# Phase-A feature spec (CircuitNet-N14)

**Status:** proposed, awaiting sign-off. Nothing in `src/` implements this yet.
**Companion to** [`progress.md`](../progress.md) §4 (the paper-derived schema). Where this
file and progress.md §4 disagree, this file describes *Phase A only* and §4 remains the
Phase-B / paper target. Section 5 below maps one onto the other.

---

## 1. What the CircuitNet timing files actually contain (verified)

Decoded against `external/CircuitNet-code/net_delay_prediction/build_graph.py`, then
verified empirically on `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap`.

We have **one design** (Vortex-small) × **96 PPA configs** × **3 stages**
(`place`, `cts`, `route`), plus one `pin_positions` file per config (stage-independent).

### `nodes.npz` → `nodes`
`(N,)` array of **pin-name strings only**. No features, no labels. Names are
backslash-escaped (`genblk1\[0\]\.cluster/...`); strip `\` to key into `pin_positions`.
Array index = node ID used by `net_edges`.

### `net_edges.npz` → `net_edges`
`(E, 6)` float64:

| Col | Meaning | Verified how |
|---|---|---|
| 0 | source (driver) pin index | used as edge src in reference builder |
| 1 | destination (sink) pin index | used as edge dst |
| 2–5 | **net delay, 4 channels** | reference builder assigns `[:,2:]` to `net_delay` |

**Graph shape is strictly bipartite driver→sink.** Measured: 117,541 nodes with
out-degree > 0, 302,895 with in-degree > 0, **zero** nodes with both. Every sink has
in-degree exactly 1. So:
- a **net** ≡ a driver pin; there are 117,541 nets in this config
- **fanout** ≡ driver out-degree (median 1, p99 31, max 1708) — derivable, no extra file
- there are **no cell/timing-arc edges** anywhere in the dataset

**The 4 delay channels** pair up as (2,3) and (4,5): corr(c2,c3)=0.9999,
corr(c4,c5)=1.0000, corr(c2,c4)=0.9884. Channel means 0.0134 / 0.0142 / 0.0134 / 0.0143.
This is consistent with TimingGCN's `[rise_early, fall_early, rise_late, fall_late]`
convention (rise/fall within a pair, early/late across pairs). **Caveat:** late ≥ early
holds for only 89% of rows and the two are exactly equal for 62%, so I would not build
any logic that assumes a corner ordering. Treat it as "4 net-delay channels", regress all
4, report per-channel metrics. 1.6% of edges are all-zero.

### `pin_positions.npz` → `pin_positions`
Dict of 1.77M pin-name → 8 floats. **Only the first 4 are ever populated**
(`x1, y1, x2, y2` bounding box); columns 4–7 are zero across the entire file. Covers more
pins than `nodes` (includes power/ground). Name resolution from `nodes` → `pin_positions`
succeeded on 100% of a 422-pin sample.

### Two alignment traps (this is the important part)

1. **Node index spaces differ between stages.** `place` has 420,652 nodes, `route` has
   424,931, and the orderings are not the same. Node `i` at place is **not** node `i` at
   route. CTS inserts buffers: 4,837 route-only pins. **Labels must be joined by pin
   name, never by index.** A naive index join silently mislabels most of the graph.
2. **The edge sets differ too.** 302,895 place edges vs 305,008 route edges; only
   276,412 name-pairs are shared (91% of place edges, 90.6% of route edges). So ~9% of
   pre-route nets have no post-route counterpart and vice versa. The loader must
   explicitly decide what to do with them — see open decision D3.

---

## 2. What the paper wants that CircuitNet does not have

| Paper feature (progress.md §4) | In CircuitNet? |
|---|---|
| net R, net C | **No** — nowhere in the dataset |
| net length | No (derivable as a bbox proxy from pin positions) |
| fanout | Yes (out-degree) |
| net RUDY | No |
| max/min/num vias | No |
| arc delay, arc input/output slew | **No** — no cell edges exist at all |
| pin-to-pin distance | Yes (from pin positions) |
| arc R, arc RC elmore | No |
| net delay (not a paper feature, but the natural stand-in target) | Yes, 4 channels, all 3 stages |

Six of the seven Level-1 node features and eight of the ten Level-2 node features are
unavailable. This is why we are **not** emitting the paper's schema with zero-padding —
the models would adapt to mostly-constant columns and Phase B would invalidate them.

---

## 3. Proposed Phase-A schema

Principle: **only fields that both CircuitNet and Cadence can populate for real.**
Everything else waits for Phase B rather than being faked now.

### Level-1 graph — net-level

- **Node** = one net (≡ its driver pin). Count ≈ 117.5k per config.
- **Node features (10):**
  `[pre_net_delay_0..3, driver_x, driver_y, bbox_w, bbox_h, fanout, log_net_bbox_hpwl]`
  — the first four are the **post-placement net delay**, log-transformed into the same
  space as the target. This mirrors the paper's Table 1, where the pre-route value of the
  predicted quantity (net R or C) is itself the first node feature: the model *corrects an
  estimate*, it does not predict from geometry alone. Measured correlation with the target
  is 0.93, so omitting them would both cripple the model and make the y=x baseline
  incomparable. The rest: driver_x/y from the driver pin bbox center; bbox_w/h and HPWL
  over the driver + all sink pins. All from `place`-stage data.
- **Edge** = net_j → net_i where their bounding boxes overlap (paper §3.3.1). Features:
  `[avg_x_i, avg_y_i, avg_x_j, avg_y_j, relative_position_quadrant]` — unchanged from
  progress.md §4, both sources can produce this.
- **Label (4):** post-route net delay, the 4 channels, log-transformed (§4 below).

Note this changes the Level-1 *target* from net R / net C to net delay. It is the same
model shape (regress a per-net vector from placement geometry + connectivity) and the
same role in the pipeline. Level-1's output is still what Level-2 consumes.

### Level-2 graph — arc-level

- **Node** = one timing arc. **CircuitNet has no cell edges**, so Phase A synthesizes
  arcs from pin names: pins sharing an instance prefix (`.../U764/A2`, `/B1`, `/Q`)
  belong to one cell; every (input pin → output pin) pair is an arc. Cadence gives real
  arcs in Phase B — the loader must emit the same structure either way so Stage 2 never
  learns it came from a heuristic.
- **Node features (5):**
  `[pin_to_pin_distance, src_x, src_y, dst_x, dst_y]`
  **+ Level-1's predicted net delay (4)** for the net driving this arc = **9 total**.
  The Level-1 predictions (not ground truth) are what make this a genuine two-stage
  pipeline — that property is preserved exactly.
- **Edge:** arcs within overlapping nets, same 5 edge features as Level-1.
- **Label (4):** post-route net delay on the arc's output net.

### Carried in `Data` but not fed to models (yet)

Every `Data` object also carries `feature_names: List[str]` and
`available_mask: Tensor[bool]`. Phase B adds R/C/RUDY/via columns by declaring them
available; `in_dim` is read from the spec, never hardcoded in a model. That is the whole
mechanism that makes the Cadence swap cheap.

---

## 4. Preprocessing decisions

- **Label transform:** `log(1e-4 + delay) + 9.211`, copied from CircuitNet's
  `data_graph.py`. Raw delays span 0 → 0.38 with a heavy mass near zero; the reference
  implementation trains in log space and so should we, for comparability.
- **Loss / metric:** MSE in log space for training (matches progress.md §7); report R²
  per channel, matching the reference's `r2_score`, plus raw-space MAE for the report.
- **Normalization:** per-config z-score on coordinates, computed on train configs only.

---

## 4b. Known limitations of the current loader

- **Unresolved pin positions.** A small number of pins in `nodes` have no `pin_positions`
  entry (measured: 558/420,652 = 0.13% on Vortex-small, up to 4,331/1,028,855 = 0.42% on
  one nvdla config). Those pins currently get position `(0, 0)`, which puts any net
  driven by such a pin at the origin. Low volume, but it is a silent wrong value rather
  than a missing one — worth dropping those nets outright in a later revision.
- **Net-level label aggregation.** Net delay is natively per driver-sink pair; Level-1
  nodes are nets, so the label is the max over the net's sinks per channel. This is the
  timing-meaningful choice but it does discard the per-sink distribution. An edge-level
  Level-1 variant (matching the reference implementation exactly) stays open.
- **Overlap graph is approximate.** Candidate pairs come from a center-binned uniform
  grid with a per-net degree cap, so the net-to-net edge set is a sampled subset of the
  true bbox-overlap relation, not the exact one.

## 5. Phase-B mapping (what `load_cadence_reports()` will fill)

| Phase-A field | Phase-B source |
|---|---|
| driver_x/y, bbox, pin-to-pin distance | Innovus DEF / placement dump |
| fanout | netlist |
| net delay labels | post-route `.rpt` / SPEF-annotated STA |
| **+ net R, net C** | post-route SPEF → re-enables the paper's true Level-1 targets |
| **+ arc delay, arc slew** | Innovus timing report → real Level-2 targets |
| **+ RUDY, via counts** | congestion / route reports (stretch, matches CNN branch) |

At that point the Level-1 target reverts from net delay to net R/C per progress.md §4,
and Level-2 gains its real labels. Model code does not change — only the spec and the
loader.

---

## 6. Decisions — RESOLVED 2026-08-11

- **D1. Pre-route input stage → `place`, label from `route`.** Honest match to the
  paper's framing. `cts → route` noted as a possible secondary experiment; not built.
- **D2. Second design → yes, one more.** Enables an unseen-*design* split in Phase A and
  keeps the loader from baking in Vortex-specific assumptions.
- **D3. Non-shared edges (~9%) → dropped for v1.** Count reported in the writeup. A
  masked-loss variant that keeps place-only edges stays open as a later refinement.
- **D4. Graph size → per-config on-disk cache in `data/graphs/` + METIS partitioning.**
  420k nodes × 96 configs will not fit in memory naively; CircuitNet's own demo uses
  `metis_partition` for exactly this.
- **D5. Library → PyTorch Geometric.** Sticks to `requirements.txt` and progress.md. The
  DGL reference is ~80 lines and ports cleanly to PyG `MessagePassing`.
