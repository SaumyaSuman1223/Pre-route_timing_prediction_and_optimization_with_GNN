"""
Loops the Genus synth_template.tcl over every circuit in benchmarks/ and every
clock-period configuration, invoking Genus in batch mode per (circuit, config) pair.
Fill in with subprocess calls to `genus -batch -files synth_template.tcl -variable ...`
once your Cadence environment/module-load setup is confirmed.
"""
