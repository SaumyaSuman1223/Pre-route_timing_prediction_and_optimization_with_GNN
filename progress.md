# PROGRESS.md — pre-route-gnn implementation status

**Read this file first, every session — human or Claude Code.** It's the single source of
truth for what's done, what's decided, and what's next. Update the checklist and the
decisions log as you go; don't let this drift out of sync with the actual code.

---

## 1. Project summary

Reproducing: Kyungjoon Chang and Taewhan Kim, **"Pre-route Timing Prediction and
Optimization with Graph Neural Network Models"**, *Integration, the VLSI Journal*, Vol.
99, 2024. DOI: 10.1016/j.vlsi.2024.102262.

**Core idea:** a two-stage, hierarchical GNN+CNN framework that predicts post-route
timing from pre-route (post-placement) data:
- **Level-1** (two GNN sub-models + one CNN sub-model): predicts net R, net C (via GNN),
  and arc-length delta / congestion (via CNN).
- **Level-2** (one GNN model): predicts arc delay and arc output slew, consuming Level-1's
  *predictions* (not ground truth) as part of its input features.

**Course context:** VLSI course, individual project, 4 deliverables over 4 months.
Professor has confirmed a well-executed 50% implementation is acceptable, but this
paper's scope (see Section 3, "Scope decisions") is realistically fully completable.

**Critical tool note:** the paper uses Synopsys ICC2 for Section 4 (feeding predictions
back into the EDA tool chain for re-optimization, i.e. their Table 7 results). We only
have Cadence. **Section 4 / Table 7 reproduction is explicitly OUT OF SCOPE.** We are
reproducing the prediction models only (their Tables 3, 4, 5 — MSE accuracy of net
R/C/arc-length/delay/slew predictions) plus one ablation (two-stage vs. single-stage
baseline). This is a deliberate, documented scope decision, not an oversight — state it
explicitly in the final report.

---

## 2. Data strategy (two phases, one schema)

**Phase A (current):** prototype and debug the whole architecture against **CircuitNet
(N14 version)**, an existing open dataset, so the model-building work isn't blocked on
Cadence pipeline setup.

**Phase B (later):** once the architecture works on CircuitNet, swap in our own
Cadence-Genus/Innovus-generated dataset (from `cadence/` + `benchmarks/`) for the final,
real results.

**The rule that makes this work without a rewrite later:** `src/data/dataset.py` must
have TWO loader functions — `load_circuitnet(...)` and `load_cadence_reports(...)` —
that both normalize into the SAME graph `Data` object schema (Section 4 below). Nothing
in `src/models/` should ever need to know which source it's training on. If you're
about to write code that only works for one data source's raw format, stop and put that
logic inside the corresponding loader instead.

### Getting CircuitNet

```bash
# 1. Clone the code repo (feature extraction, build_graph.py, tutorials)
git clone https://github.com/circuitnet/CircuitNet.git external/CircuitNet-code

# 2. Get the actual data — CircuitNet-N14, via Hugging Face (easiest, avoids
#    Google Drive/Baidu throttling). N14 is the right version for us specifically
#    because it has features extracted at THREE stages (post-placement, post-CTS,
#    post-route) — post-placement = pre-route, matching this paper's exact framing.
pip install huggingface_hub
huggingface-cli download CircuitNet/CircuitNet --repo-type dataset --local-dir data/external/circuitnet

# 3. We specifically need the TIMING feature tarballs: nodes.tar.gz, net_edges.tar.gz,
#    pin_positions.tar.gz (net_edges has 6 features per edge incl. source/dest node
#    index — this is what becomes our graph structure + delay labels).
#    Extract these into data/external/circuitnet/timing/

# 4. Build graphs using CircuitNet's own script as a REFERENCE (don't depend on it
#    long-term — we want our own load_circuitnet() in src/data/dataset.py using the
#    same underlying files, output into OUR schema, not theirs):
python external/CircuitNet-code/build_graph_demo/build_graph.py \
    --data_path data/external/circuitnet/timing --save_path data/graphs/circuitnet_raw
```

If Hugging Face download is slow/unavailable, fall back to the Google Drive/Baidu
Netdisk links on https://circuitnet.github.io/intro/download.html — same data, slower
to fetch.

---

## 3. Scope decisions (append-only — don't rewrite past decisions, add new ones below)

- Reproducing Level-1 (net R, net C, arc-length delta) + Level-2 (arc delay, arc slew)
  prediction models. NOT reproducing Section 4 (Synopsys ICC2 timing-optimization
  loop) — tool mismatch, documented above.
- Benchmark circuits: open OpenCores designs + Nangate 15nm PDK (matches the paper's
  own setup, both freely available) — for our own Cadence-generated data in Phase B.
  8–12 circuits, several clock-period configs each.
- Prototyping architecture against CircuitNet-N14 before Cadence pipeline is ready
  (this file's Section 2), to de-risk model-building from data-pipeline-building.
- CNN routing-resource branch (Section 3.4 of the paper) is a STRETCH GOAL, not core —
  attempt only after the GNN-only pipeline (Level-1 R/C via GNN, Level-2 via GNN) works
  end-to-end.
- Core ablation: two-stage pipeline vs. a single-stage baseline that predicts
  delay/slew directly from pre-route features in one shot (roughly reproducing Guo et
  al.'s TimingGCN framing) — this is what actually demonstrates the paper's
  contribution, not just "built a GNN that predicts timing."

---

## 4. Data schema (the contract between loaders and models — DO NOT change casually;
if you must change it, update every loader AND every model at the same time)

**Level-1 net-based graph** (separate instances for R-prediction and C-prediction):
- Node: one net. Features: `[R_or_C, net_length, fanout, net_RUDY, max_via, min_via,
  num_vias]` (7 features, Table 1 / Section 3.3.1–3.3.2).
- Edge: directed, from net_j to net_i if bounding boxes overlap. Features:
  `[avg_x_i, avg_y_i, avg_x_j, avg_y_j, relative_position_quadrant]` (5 features).

**Level-1 CNN (congestion) input:**
- 3D tensor, `5 x W x H` per net (G-cell RUDY map, horizontal net density, vertical net
  density, source/sink location map, pin-capacitance location map), adaptive-pooled to
  `5 x 7 x 7`.

**Level-2 arc-based graph:**
- Node: one arc. Features: `[arc_delay, arc_input_slew, arc_output_slew,
  pin_to_pin_distance, arc_R, arc_RC_elmore, arc_max_via, arc_min_via, net_R, net_C]`
  (10 features — net_R/net_C come from Level-1's PREDICTIONS, not ground truth).
- Edge: directed, arcs within overlapping nets. Features:
  `[avg_x_i, avg_y_i, avg_x_j, avg_y_j, relative_position_quadrant]` (same shape as
  Level-1 edges).

**Labels:** post-route net R, net C, arc length (Level-1); post-route arc delay, arc
output slew (Level-2). All regression targets, MSE-evaluated (paper's Tables 3–5).

**CircuitNet mapping note:** CircuitNet's `net_edges` array gives 6 channels including
source/destination node index — map these into the schema above; CircuitNet won't have
every paper-specific feature (e.g. via-layer counts, RUDY) out of the box for all
stages, so some fields may be zero/placeholder during Phase A prototyping and only
become real once we're on our own Cadence data in Phase B. Note explicitly in code
comments wherever a feature is a Phase-A placeholder.

> **UPDATE (2026-08-11) — see [`docs/phase_a_feature_spec.md`](docs/phase_a_feature_spec.md).**
> The `net_edges` format is now decoded and verified: cols 0–1 are src/dst pin index,
> cols 2–5 are 4-channel net delay. The gap is much larger than "some fields may be
> zero": CircuitNet has **no net R, no net C, no RUDY, no via counts, and no cell/arc
> edges at all**. Rather than zero-pad the schema above (which would train the models on
> mostly-constant columns and invalidate them at the Phase-B swap), Phase A uses a
> reduced, honestly-populated feature set with net *delay* as the Level-1 target. The
> spec doc defines it and maps it back onto this section for Phase B. Section 4 here
> remains the paper/Phase-B target.

---

## 5. Paper section → code file map

| Paper section | What it specifies | Code location |
|---|---|---|
| Table 1 | Full feature list per sub-model | `src/data/dataset.py` |
| §3.3.1 / §3.3.2, Fig. 10–11 | Net-based graph construction for R and C | `src/data/netlist_parser.py` |
| Eq. (2)–(5) | GAT layer math | `src/models/net_arc_conv.py` |
| §3.3 | Level-1 GNN (3 layers, R and C sub-models, separate training) | `src/models/stage1_model.py` |
| §3.4, Fig. 12 | CNN congestion/arc-length model | `src/models/` (STRETCH GOAL — not yet scoped a file) |
| §3.5, §3.5.1, Fig. 13 | Arc-based graph + Level-2 GNN (delay, slew) | `src/models/stage2_model.py` |
| §4 | EDA tool-chain re-optimization (Synopsys ICC2) | **OUT OF SCOPE — not implemented** |
| Table 2 | Benchmark circuits (OpenCores + Nangate 15nm) | `benchmarks/` |
| §5.1 | "Seen"/"unseen" train/test protocol | `src/train.py` |

---

## 6. Status checklist

**Phase A — CircuitNet prototyping**
- [x] CircuitNet-N14 downloaded, timing tarballs extracted — Vortex-small (96 usable
      configs) and nvdla-small (77 usable; the tarball itself is missing route and
      pin_position files for 19). **173 usable graphs.** See
      [`docs/dataset_summary.md`](docs/dataset_summary.md).
- [x] `build_graph.py` decoded as a reference, feature semantics verified — see
      [`docs/phase_a_feature_spec.md`](docs/phase_a_feature_spec.md)
- [x] Phase-A feature spec signed off (decisions D1–D5 resolved 2026-08-11)
- [x] `src/data/dataset.py`: `load_circuitnet()` implemented (in `circuitnet.py`, re-exported),
      normalized to the Phase-A schema in `src/data/feature_spec.py`. Verified on both
      designs: ~15s/config, avg degree 8.3–8.6, 2.5–3.5% isolated nets, no NaNs.
- [x] Cache all 173 configs to `data/graphs/` (`python -m src.data.build_cache`, ~6s each,
      ~55 MB per graph, ~10 GB total)
- [x] `src/models/net_arc_conv.py`: GAT layer with edge-feature-conditioned attention
      (paper Eq. 2–5), shared by both levels
- [x] `src/data/partition.py`: spatial tiling — no METIS backend is installed, and
      bbox-overlap edges are spatially local anyway (90.7% edge retention at 16 tiles)
- [x] `src/models/stage1_model.py` + `src/train.py`: trains and evaluates on CircuitNet,
      both protocols, always reported against the identity baseline
- [ ] Full-cache retrain (the result below used 31 of 96 Vortex configs — cache was
      still building) + the `unseen-design` protocol against nvdla-small
- [ ] `src/models/stage2_model.py`: needs arc-graph construction first (synthesize cell
      arcs from instance prefixes in pin names — CircuitNet ships no cell edges)
- [ ] `src/models/baseline_single_stage.py`: implemented, evaluated for the core ablation

### First Level-1 result (2026-08-11)

`python -m src.train --designs Vortex-small --epochs 25 --hidden 128 --heads 4 --parts 16`
23 train / 4 val / 4 test graphs, 88,324 parameters, ~13s/epoch on an RTX 3050.

| | model | identity baseline | delta |
|---|---|---|---|
| R² (mean of 4 channels) | **0.9129** | 0.8654 | **+0.0475** |
| MAE (log space) | **0.2807** | 0.2955 | **−0.0148** |

Per-channel R²: 0.920 / 0.918 / 0.908 / 0.906 vs baseline 0.873 / 0.875 / 0.856 / 0.858.
Test set = 469,128 held-out nets. The model beats the pre-route estimate on every
channel, on both metrics. Not yet a paper-level claim — one design, unseen-config only.

**Phase B — real Cadence data**
- [ ] `learning/01`–`03` mini-projects complete (STA basics, synthesis sweep, pre/post-route comparison)
- [ ] Benchmark circuits selected, RTL in `benchmarks/`
- [ ] `cadence/genus/` + `cadence/innovus/` scripts working for one circuit end-to-end
- [ ] Scripted across full benchmark set x config sweep
- [ ] `src/data/dataset.py`: `load_cadence_reports()` implemented, normalized to same schema
- [ ] Full retrain + evaluation on real data
- [ ] Final ablation results (two-stage vs. single-stage) on real data
- [ ] `report/final_report.md` written

---

## 7. Open questions / things the paper doesn't specify (fill in as resolved)

- Training hyperparameters (learning rate, optimizer, epochs, batch size) — NOT stated
  in the paper. Decide empirically; log final choices here once settled.
  *(In use so far: Adam, lr 1e-3, MSE in log space, grad-clip 1.0, 3 layers, 4 heads,
  hidden 64–128, one spatial tile per optimizer step. Head count and hidden width are
  also unstated by the paper.)*
- **The identity baseline is the number to beat.** Using the pre-route estimate unchanged
  already gives R² ≈ 0.84 / MAE ≈ 0.31 on this data. Any Level-1 result must be quoted
  next to it — `src/train.py` computes both on the same nets, every evaluation.
- Exact training loss function — implied MSE (matches their eval metric) but never
  explicitly stated as the training objective. Using MSE unless a reason emerges not to.
- XGBoost margin-function (A, B, C) training details are thin in the paper (§4) — not
  needed since Section 4 is out of scope for us.

---

## 7b. How to run it

All commands, in order, with troubleshooting: [`docs/RUNNING.md`](docs/RUNNING.md).
Short version — `python -m src.data.build_cache` once, then
`python -m src.train --designs Vortex-small --epochs 25 --hidden 128 --parts 16`.

---

## 8. For Claude Code / next session, start here

1. Read this file in full, then `README.md`.
2. Check `learning/` — which mini-projects (01–08) have actual code vs. just the README stub.
3. Check `src/` — which modules are still docstring-only stubs vs. real implementations.
4. Check Section 6's checklist above for the next unchecked item — work top to bottom,
   Phase A before Phase B, unless the user says otherwise.
5. Before writing new logic, check Section 4 (schema) — don't invent a different node/
   edge feature layout without updating this file and every consumer of the schema.
6. When a checklist item is completed, check it off in this file in the same session —
   don't leave the checklist stale.