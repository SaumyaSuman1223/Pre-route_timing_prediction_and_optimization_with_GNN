# 03 — Pre-route vs. post-route timing comparison

Goal: place (no route) one synthesized netlist, pull a timing report, then fully route it
and pull a second timing report. Compare the same critical path's delay before vs. after
real routing exists. This IS the (input, label) pair the whole paper predicts — this
mini-project produces your first real training example, by hand.

Files:
- innovus_place.tcl
- innovus_route.tcl
- preroute_report.rpt   (generated)
- postroute_report.rpt  (generated)
- comparison.md         — your written before/after comparison + explanation
