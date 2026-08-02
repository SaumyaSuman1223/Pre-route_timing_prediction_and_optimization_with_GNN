"""
PyTorch Geometric Dataset wrapping netlist_parser.py + report_parser.py output.
Loads (or builds and caches to data/graphs/) one graph per (circuit, config) pair,
with net-edge / arc-edge typed edges and preroute-derived features / postroute-
derived labels attached.
"""
