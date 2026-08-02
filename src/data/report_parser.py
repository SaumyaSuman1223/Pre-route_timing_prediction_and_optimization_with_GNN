"""
Parses a Cadence Innovus timing report (.rpt) into structured (feature, label) data:
  - from a PREROUTE report: net R, net C estimates, arc length, fanout, cell type
    -> these become node/edge FEATURES
  - from a POSTROUTE report: real net R, real net C, real arc delay, real slack
    -> these become node/edge LABELS

Used by both cadence/innovus/run_all_par.py (to sanity-check reports as they're
generated) and src/data/dataset.py (to build the training set).
"""
