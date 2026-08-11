"""
Declared feature schema shared by every loader and every model.

This is the mechanism that makes the Phase-A (CircuitNet) -> Phase-B (Cadence) swap a
loader change rather than a rewrite. See docs/phase_a_feature_spec.md.

The rule: a loader may only emit columns it can populate with REAL values. Fields that a
data source cannot provide are declared here with phase_a=False and are simply absent
from the tensor that source produces -- they are never zero-padded, because a model
trained on a mostly-constant column adapts to that column and is invalidated when the
column becomes real.

Models read their input width from `active_dim(...)`, never from a hardcoded integer.
When Phase B lands, `load_cadence_reports()` passes phase='B', more fields become
active, `active_dim` grows, and the models are retrained -- with the difference in the
feature set explicitly recorded rather than silently absorbed.
"""

from dataclasses import dataclass
from typing import List, Literal, Sequence

Phase = Literal["A", "B"]


@dataclass(frozen=True)
class Field:
    """One column of a node- or edge-feature tensor."""

    name: str
    #: where the value comes from, for documentation and for Phase-B wiring
    source: str
    #: True if CircuitNet-N14 can populate this field with a real (non-placeholder) value
    phase_a: bool
    #: what the field means, and any caveat about how it is derived
    note: str = ""


# --------------------------------------------------------------------------------------
# Level 1 -- net-level graph. One node per net (== per driver pin).
# --------------------------------------------------------------------------------------

LEVEL1_NODE: List[Field] = [
    # The PRE-ROUTE estimate of the quantity being predicted. The paper's Table 1 lists
    # net R (or C) as the first Level-1 node feature and the post-route value as the
    # label -- i.e. the model CORRECTS an estimate rather than predicting from geometry
    # alone. These four channels are the Phase-A equivalent (post-placement net delay),
    # and they are why the y=x baseline is the number to beat.
    Field("pre_net_delay_0", "sta", True, "post-placement net delay, channel 0, log-space"),
    Field("pre_net_delay_1", "sta", True, "post-placement net delay, channel 1, log-space"),
    Field("pre_net_delay_2", "sta", True, "post-placement net delay, channel 2, log-space"),
    Field("pre_net_delay_3", "sta", True, "post-placement net delay, channel 3, log-space"),
    Field("driver_x", "geometry", True, "driver pin bbox center x"),
    Field("driver_y", "geometry", True, "driver pin bbox center y"),
    Field("net_bbox_w", "geometry", True, "width of bbox over driver + all sink pins"),
    Field("net_bbox_h", "geometry", True, "height of bbox over driver + all sink pins"),
    Field("fanout", "connectivity", True, "driver out-degree; nets are strictly bipartite"),
    Field("log_hpwl", "geometry", True, "log1p(net_bbox_w + net_bbox_h)"),
    # ---- Phase B only: require parasitics / routing reports Cadence gives us ----
    Field("pre_net_R", "parasitics", False, "pre-route R estimate; the paper's Table-1 slot"),
    Field("pre_net_C", "parasitics", False, "pre-route C estimate; the paper's Table-1 slot"),
    Field("net_rudy", "routing", False, "congestion report"),
    Field("max_via", "routing", False, "route report"),
    Field("min_via", "routing", False, "route report"),
    Field("num_vias", "routing", False, "route report"),
]

# Unchanged from progress.md section 4 -- both data sources can produce all of these.
LEVEL1_EDGE: List[Field] = [
    Field("avg_x_i", "geometry", True, "center x of net i"),
    Field("avg_y_i", "geometry", True, "center y of net i"),
    Field("avg_x_j", "geometry", True, "center x of net j"),
    Field("avg_y_j", "geometry", True, "center y of net j"),
    Field("rel_quadrant", "geometry", True, "quadrant of j relative to i, 0-3"),
]

# --------------------------------------------------------------------------------------
# Level 2 -- arc-level graph. One node per timing arc.
# --------------------------------------------------------------------------------------

LEVEL2_NODE: List[Field] = [
    Field("pin_to_pin_dist", "geometry", True, "input pin -> output pin distance"),
    Field("src_x", "geometry", True, "arc input pin x"),
    Field("src_y", "geometry", True, "arc input pin y"),
    Field("dst_x", "geometry", True, "arc output pin x"),
    Field("dst_y", "geometry", True, "arc output pin y"),
    # Stage-1 PREDICTIONS, not ground truth. This is what makes the pipeline two-stage.
    Field("pred_net_delay_0", "stage1_pred", True, "Level-1 output, channel 0"),
    Field("pred_net_delay_1", "stage1_pred", True, "Level-1 output, channel 1"),
    Field("pred_net_delay_2", "stage1_pred", True, "Level-1 output, channel 2"),
    Field("pred_net_delay_3", "stage1_pred", True, "Level-1 output, channel 3"),
    # ---- Phase B only ----
    Field("arc_delay", "sta", False, "Innovus timing report"),
    Field("arc_input_slew", "sta", False, "Innovus timing report"),
    Field("arc_output_slew", "sta", False, "Innovus timing report"),
    Field("arc_R", "parasitics", False, "post-route SPEF"),
    Field("arc_rc_elmore", "parasitics", False, "post-route SPEF"),
    Field("arc_max_via", "routing", False, "route report"),
    Field("arc_min_via", "routing", False, "route report"),
    Field("pred_net_R", "stage1_pred", False, "Level-1 prediction once R is a target"),
    Field("pred_net_C", "stage1_pred", False, "Level-1 prediction once C is a target"),
]

LEVEL2_EDGE: List[Field] = list(LEVEL1_EDGE)

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

#: CircuitNet net_edges[:, 2:6]. Verified: channels pair as (0,1) and (2,3), consistent
#: with TimingGCN's [rise_early, fall_early, rise_late, fall_late]. The corner ordering is
#: NOT reliable (late >= early holds for only 89% of rows), so nothing may depend on it --
#: all four are regressed and reported per-channel.
NET_DELAY_CHANNELS = 4

#: Label transform, copied from CircuitNet's data_graph.py so our numbers stay comparable
#: to the published reference. log(1e-4) ~= -9.2103.
LOG_OFFSET = 1e-4
LOG_SHIFT = 9.211


def active(fields: Sequence[Field], phase: Phase = "A") -> List[Field]:
    """The fields a loader may actually emit in the given phase, in column order."""
    return [f for f in fields if phase == "B" or f.phase_a]


def active_names(fields: Sequence[Field], phase: Phase = "A") -> List[str]:
    return [f.name for f in active(fields, phase)]


def active_dim(fields: Sequence[Field], phase: Phase = "A") -> int:
    """Input width for a model. Never hardcode this number in src/models/."""
    return len(active(fields, phase))
