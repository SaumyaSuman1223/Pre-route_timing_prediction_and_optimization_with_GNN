# 02 — Synthesis design-space mini-sweep

Goal: synthesize 2-3 small circuits at multiple clock constraints, tabulate area/slack/
cell-count, and observe how tightening timing changes gate selection. This script is the
direct ancestor of cadence/genus/run_all_synth.py.

Files:
- circuits/            — small Verilog circuits used in the sweep
- genus_sweep.tcl       — parametrized synthesis script (takes a clock period arg)
- run_sweep.py          — loops the tcl script over circuits x constraints
- results_table.csv     — collected area/slack/cell-count results
