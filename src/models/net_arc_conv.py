"""
Custom MessagePassing layer with separate message functions for NET edges vs.
ARC edges.

Promoted from: learning/06_edge_type_message_passing/custom_message_passing.py

This is the paper's core technique (Chang & Kim's "net and arc GNN"), now operating
on real circuit graphs instead of synthetic two-edge-type toy graphs.
"""
