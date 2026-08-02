# Parametrized Genus synthesis script.
# Expected variables (set via -variable at invocation, or by run_all_synth.py):
#   CIRCUIT_NAME   - top module name
#   RTL_FILE       - path to Verilog source
#   CLOCK_PERIOD   - target clock period in ns
#   OUT_DIR        - where to write the synthesized netlist + reports
#
# read_hdl $RTL_FILE
# elaborate $CIRCUIT_NAME
# read_sdc <matching constraints/*.sdc for CLOCK_PERIOD>
# syn_generic; syn_map; syn_opt
# write_hdl > $OUT_DIR/$CIRCUIT_NAME.v
# report_timing > $OUT_DIR/$CIRCUIT_NAME_synth_timing.rpt
# report_area   > $OUT_DIR/$CIRCUIT_NAME_area.rpt
