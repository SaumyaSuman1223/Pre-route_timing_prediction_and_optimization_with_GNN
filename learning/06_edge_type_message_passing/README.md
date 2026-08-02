# 06 — Custom edge-type-aware message passing

Goal: build a synthetic graph with two distinct edge types and a task whose correct
answer depends on treating them differently. Implement a custom MessagePassing subclass
that does so, and show it beats an edge-type-naive baseline. This is a rehearsal of the
paper's actual core technique (net-edges vs. arc-edges).

Files:
- synthetic_two_edge_graph.py
- custom_message_passing.py  — promoted later into src/models/net_arc_conv.py
- train_toy.py
