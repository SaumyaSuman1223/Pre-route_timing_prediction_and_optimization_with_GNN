"""
Stage 2: predicts arc delay and endpoint slack, consuming STAGE 1's PREDICTED
R/C/arc-length values as input (not the ground-truth postroute values) -- this is
what makes it a genuine two-stage pipeline rather than two independent models.

Promoted from: learning/07_two_stage_pipeline/stage2_toy.py
"""
