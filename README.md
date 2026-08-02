# pre-route-gnn

Reproduction of Chang & Kim, "Pre-route Timing Prediction and Optimization with Graph Neural Network Models" (*Integration, the VLSI Journal*, Vol. 99, 2024), scoped for individual, single-semester implementation.

## Repo philosophy: learning/ feeds src/, never duplicates it

This repo has two kinds of code, and they are related on purpose:

- **`learning/`** — small, standalone mini-projects (01 through 08), each one proving out a single concept on toy/synthetic data before it's used for real. These stay in the repo permanently as a visible record of the learning process — don't delete them once you've "graduated" past them.
- **`src/`** — the real, reusable package used by the actual paper reproduction. Code here is *promoted* from `learning/` once a mini-project has proven the idea works: take the working logic, clean it up, generalize it (real netlists instead of toy ones, real edge types instead of synthetic ones), and place it in `src/`. The `learning/` version and the `src/` version are allowed to look similar — that's fine, that's the point — but `src/` is the one everything downstream actually imports.

Concretely:
- `learning/04_netlist_to_graph/parse_netlist.py` (toy adder) → promoted into → `src/data/netlist_parser.py` (real benchmark circuits)
- `learning/06_edge_type_message_passing/custom_message_passing.py` (synthetic 2-edge-type graph) → promoted into → `src/models/net_arc_conv.py` (real net-edges vs. arc-edges)
- `learning/07_two_stage_pipeline/` (synthetic causal chain) → promoted into → `src/models/stage1_model.py` + `src/models/stage2_model.py`

## Directory map

```
pre-route-gnn/
├── learning/                  # mini-projects 1-8, kept permanently, never deleted
├── cadence/                   # Genus + Innovus TCL scripts + Python runner scripts
├── benchmarks/                # input RTL for chosen open benchmark circuits
├── data/                      # generated timing reports + parsed graphs (mostly gitignored)
├── src/                       # the real, reusable package
│   ├── data/                  # netlist parsing, report parsing, PyG Dataset
│   └── models/                # net_arc_conv, stage1, stage2, single-stage baseline
├── notebooks/                 # exploratory analysis + plots for the report
├── results/                   # figures, metrics, trained model checkpoints
└── report/                    # final written report + scope-reduction proposal
```

See the `README.md` inside each top-level folder for what belongs there.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Cadence Genus/Innovus are assumed available on your college's license server — see `cadence/README.md` for how the scripts expect to be invoked.

## Status

- [ ] Mini-projects 1-7 complete (learning/)
- [ ] Benchmark circuits selected (benchmarks/)
- [ ] Data generation pipeline scripted (cadence/)
- [ ] Netlist-to-graph pipeline (src/data/)
- [ ] Stage 1 model (net R / C / arc-length)
- [ ] Stage 2 model (delay / slack)
- [ ] Single-stage baseline for ablation
- [ ] Final report
