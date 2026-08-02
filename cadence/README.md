# cadence/

Scripts that drive Cadence Genus (synthesis) and Innovus (placement + routing + STA
extraction) across the FULL benchmark set x configuration sweep, for real dataset
generation. These are the scaled-up versions of learning/02 and learning/03.

- genus/synth_template.tcl   — parametrized synthesis script (circuit + clock period args)
- genus/run_all_synth.py     — loops synth_template.tcl over all benchmarks/ x configs
- innovus/place_template.tcl
- innovus/route_template.tcl
- innovus/run_all_par.py     — loops place+route+report over all synthesized netlists
- constraints/*.sdc          — one clock-constraint file per configuration
