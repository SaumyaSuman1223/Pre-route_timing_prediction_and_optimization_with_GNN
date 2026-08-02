# 04 — Netlist-to-graph parser (toy scale)

Goal: parse the adder's gate-level netlist into a graph (node = cell/pin, edge = net),
visualize it with networkx, and manually verify it matches the real circuit.

Files:
- parse_netlist.py      — the parser (promoted later into src/data/netlist_parser.py)
- visualize_graph.py    — networkx draw + manual verification
