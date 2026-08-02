"""
Netlist -> graph parser, real-scale version.

Promoted from: learning/04_netlist_to_graph/parse_netlist.py

Differences from the toy version:
- handles arbitrary benchmark circuits, not just the toy adder
- distinguishes NET edges from ARC (timing-arc) edges explicitly, since
  src/models/net_arc_conv.py needs that distinction at training time
- pulls node/edge features from the paired preroute .rpt (see report_parser.py)
  instead of leaving them unset
"""
