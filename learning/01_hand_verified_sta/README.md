# 01 — Hand-verified STA

Goal: synthesize a tiny 4-bit ripple-carry adder in Genus, pull the critical path's
timing report, and manually recompute its arrival time gate-by-gate to confirm you
understand what the tool is reporting.

Files:
- adder4bit.v          — the toy circuit
- genus_script.tcl      — synthesis script
- timing_report.rpt     — Genus report_timing output (generated)
- notes.md              — your hand calculation, written out
