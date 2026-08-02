# 07 — Two-stage pipeline on synthetic data

Goal: generate synthetic data with a known causal chain (input -> noisy intermediate ->
final target depending on the intermediate), train Stage 1 to predict the intermediate,
train Stage 2 on STAGE 1's PREDICTED intermediate (not the true one), and observe how
Stage 1 error propagates into Stage 2 accuracy.

Files:
- synthetic_causal_chain.py
- stage1_toy.py
- stage2_toy.py
- error_propagation_notes.md
