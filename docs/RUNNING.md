# How to run everything

Commands in order. Run from the repo root. Every command is resumable — nothing has to
be done in one sitting.

Current state on this machine: Vortex-small is fully cached (96/96), **nvdla-small is at
23/77** and needs step 3 finished before the `unseen-design` protocol will run.

---

## 0. Environment (once)

```bash
python -m venv venv && source venv/bin/activate
```

Install the torch build that matches your driver **first** (from pytorch.org), then:

```bash
pip install -r requirements.txt
```

Check the GPU is visible — training falls back to CPU silently otherwise, and it is
roughly 20x slower:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Data (once, ~10 GB download)

Already done on this machine; here for reproducibility.

```bash
hf download CircuitNet/CircuitNet --repo-type dataset --include "CircuitNet-N14/timing_features/Vortex-small.tar.gz" --local-dir data/external/circuitnet
```

```bash
tar -zxf data/external/circuitnet/CircuitNet-N14/timing_features/Vortex-small.tar.gz -C data/external/circuitnet/timing_extracted
```

nvdla-small is the second design (6.6 GB). Only `place`, `route` and `pin_positions` are
needed — skipping `cts` saves about a third:

```bash
tar -zxf data/external/circuitnet/CircuitNet-N14/timing_features/nvdla-small.tar.gz -C data/external/circuitnet/timing_extracted --wildcards './nvdla-small/place/*' './nvdla-small/route/*' './nvdla-small/pin_positions/*'
```

## 2. Look at the data (optional, ~2 min)

```bash
python -m src.visualize_dataset --configs 2
```

```bash
python -m src.dataset_stats --configs 4
```

Figures land in `results/figures/`. Both are CPU-only.

## 3. Build the graph cache (~25 min CPU, ~10 GB disk)

**Run this before training.** It parses each config once so training does not re-parse
every epoch. It skips configs already cached, so re-running resumes where it stopped —
this is the command that finishes your nvdla-small cache.

```bash
python -m src.data.build_cache
```

Roughly 6s per Vortex config and 20s per nvdla config, ~55 MB per graph.

## 4. Train Level-1

Same design, disjoint PPA configurations:

```bash
python -m src.train --designs Vortex-small --epochs 25 --hidden 128 --heads 4 --parts 16
```

Train on one design, test on a design never seen (needs step 3 finished for both):

```bash
python -m src.train --protocol unseen-design --epochs 25 --hidden 128 --heads 4 --parts 16
```

About 13s/epoch on an RTX 3050 6GB with 23 training graphs; scales roughly linearly with
graph count. Checkpoints and a per-epoch JSON history go to `results/trained_models/`.

Every line prints the model **and** the identity baseline (the pre-route estimate used
unchanged) on the same nets. The baseline scores R² ≈ 0.87 by itself, so read the `+`
delta, not the absolute number.

---

## If you hit trouble

**"cache is incomplete"** — training refuses to run on a half-built cache, because it
would quietly report results for a fraction of the data. Finish step 3, or add
`--allow-partial-cache` to proceed deliberately.

**CUDA out of memory** — raise `--parts` (more, smaller tiles) before lowering `--hidden`:

```bash
python -m src.train --designs Vortex-small --epochs 25 --hidden 128 --parts 36
```

Tiles are spatial, so raising `--parts` costs a little edge retention: 16 tiles keeps
90% of edges, 25 tiles keeps 84%. The run prints the retention it actually achieved.

**System RAM is the other limit** (this machine has 7 GB). One cached graph is ~55 MB but
inflates in memory during partitioning; `--max-train` caps how many graphs are used at
all, and `--max-eval` caps val/test.

**Slow epochs** — the bottleneck is CPU-side loading and tiling, not the GPU. Fewer,
larger tiles (`--parts 9`) speeds things up if memory allows.

## Useful flags

| Flag | Default | What it does |
|---|---|---|
| `--protocol` | `unseen-config` | `unseen-config` or `unseen-design` |
| `--designs` | both | which designs to include |
| `--epochs` | 30 | training epochs |
| `--parts` | 16 | spatial tiles per graph — the main memory knob |
| `--hidden` / `--heads` / `--layers` | 64 / 4 / 3 | model size (paper fixes 3 layers) |
| `--lr` | 1e-3 | Adam learning rate |
| `--max-train` / `--max-eval` | 0 (all) / 8 | cap graphs, for quick runs |
| `--allow-partial-cache` | off | train on an unfinished cache |
