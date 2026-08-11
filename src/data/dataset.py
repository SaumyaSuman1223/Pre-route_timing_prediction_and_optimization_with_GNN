"""
PyTorch Geometric dataset layer.

Public loaders (progress.md section 2's rule: both normalize into the SAME schema, and
nothing in src/models/ ever learns which source it is training on):

  load_circuitnet(...)        Phase A -- CircuitNet-N14, implemented in circuitnet.py
  load_cadence_reports(...)   Phase B -- Genus/Innovus reports, not yet implemented

The schema contract itself lives in feature_spec.py.
"""

from .circuitnet import load_circuitnet  # noqa: F401


def load_cadence_reports(*args, **kwargs):
    """Phase B. See docs/phase_a_feature_spec.md section 5 for the field mapping."""
    raise NotImplementedError("Phase B: waiting on the Cadence-generated dataset")
