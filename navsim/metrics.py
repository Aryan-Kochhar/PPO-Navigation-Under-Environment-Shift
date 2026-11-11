"""Shared metric code: freezing detector, failure taxonomy, episode metrics.

This module is the single source of truth for the labels. Study A, Study B, the
baselines and the demo all call ``classify`` / ``episode_metrics`` from here, with
the thresholds in ``config.py``, so every number in every table is computed the
same way.

Taxonomy (ordered tests, first match wins; applied only to failed episodes):
    1. collision_from_behind
    2. freezing
    3. oscillation_near_goal
    4. corridor_deadlock
    5. timeout
    6. frontal_static_collision   (amendment, added after the pilot run and
                                   before the full experiment; see config.py)
    -> residual  (no test fires; reported, never repaired)
"""

import math

import numpy as np

from . import config as C


class EpisodeRecorder:
    """Collects the per-step diagnostics an env exposes in ``env.diag``."""

    KEYS = ["progress", "a_free", "omega", "v", "dist_goal", "x", "y", "clear_l", "clear_r", "clearance"]

    def __init__(self):
        self.buf = {k: [] for k in self.KEYS}
        self.outcome = None
        self.contact = None

    def record(self, env, info):
        for k in self.KEYS:
            self.buf[k].append(env.diag[k])
        if info.get("outcome"):
            self.outcome = info["outcome"]
            self.contact = info.get("contact")

    def arrays(self):
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.buf.items()}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _has_run(mask, n):
    """True if ``mask`` contains a run of >= n consecutive True values."""
    if n <= 0:
        return bool(mask.any())
    if len(mask) < n:
        return False
    c = np.convolve(mask.astype(np.int32), np.ones(n, dtype=np.int32), mode="valid")
    return bool((c >= n).any())


def freezing_condition(rec, delta=C.FREEZE_DELTA):
    """Per-step freezing condition: no progress AND a clear action exists."""
    return (rec["progress"] < delta) & (rec["a_free"] >= 1)


def has_freezing_event(rec, delta=C.FREEZE_DELTA, n=C.FREEZE_N):
    return _has_run(freezing_condition(rec, delta), n)


def is_collision_from_behind(contact):
    """Contact with a dynamic obstacle that was outside the sensor FOV, or that sat
    behind the robot's lateral axis while moving along the robot's heading."""
    if not contact or contact.get("kind") != "dynamic":
        return False
    b = contact["bearing"]
    if np.isnan(b):
        return False
    outside_fov = abs(b) > C.LIDAR_FOV / 2
    behind_and_closing = abs(b) > math.pi / 2 and contact["closing_along_heading"]
    return bool(outside_fov or behind_and_closing)


def has_oscillation_near_goal(rec):
    w = rec["omega"]
    s = np.where(np.abs(w) < C.OSC_DEADBAND, 0, np.sign(w))
    # a flip is a nonzero sign different from the last nonzero sign
    flips = np.zeros(len(s), dtype=bool)
    last = 0
    for i, si in enumerate(s):
        if si != 0:
            if last != 0 and si != last:
                flips[i] = True
            last = si
    near = rec["dist_goal"] < C.OSC_RADIUS
    flips &= near
    if len(flips) < C.OSC_WINDOW:
        return bool(flips.sum() >= C.OSC_MIN_FLIPS)
    c = np.convolve(flips.astype(np.int32), np.ones(C.OSC_WINDOW, dtype=np.int32), mode="valid")
    return bool((c >= C.OSC_MIN_FLIPS).any())


def has_corridor_deadlock(rec):
    n = C.DEADLOCK_WINDOW
    cond = (rec["clear_l"] < C.DEADLOCK_LATERAL) & (rec["clear_r"] < C.DEADLOCK_LATERAL) & (rec["a_free"] == 0)
    if len(cond) < n:
        return False
    c = np.convolve(cond.astype(np.int32), np.ones(n, dtype=np.int32), mode="valid")
    starts = np.nonzero(c >= n)[0]
    for s in starts:
        e = s + n - 1
        if math.hypot(rec["x"][e] - rec["x"][s], rec["y"][e] - rec["y"][s]) < C.DEADLOCK_EPS:
            return True
    return False


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #
def classify(rec, outcome, contact):
    """Return the taxonomy label. ``success`` for successful episodes."""
    if outcome == "success":
        return "success"
    if outcome == "collision" and is_collision_from_behind(contact):
        return "collision_from_behind"
    if has_freezing_event(rec):
        return "freezing"
    if has_oscillation_near_goal(rec):
        return "oscillation_near_goal"
    if has_corridor_deadlock(rec):
        return "corridor_deadlock"
    if outcome == "timeout":
        return "timeout"
    if outcome == "collision":
        # any contact not caught by test 1: head-on / side contact with a moving
        # obstacle, or contact with a wall or static block
        return "frontal_static_collision"
    return "residual"


def collision_kind(outcome, contact):
    """Descriptive sub-type of a collision (not part of the taxonomy; used to
    break down the frontal_static_collision category in the analysis)."""
    if outcome != "collision" or not contact:
        return ""
    if contact["kind"] != "dynamic":
        return contact["kind"]                       # static / wall
    return "dynamic_behind" if is_collision_from_behind(contact) else "dynamic_front"


def sens_key(dfrac, n):
    return f"sens_d{dfrac:.2f}_n{n}"


def episode_metrics(recorder, steps, path_length):
    """All per-episode numbers that go into the CSVs."""
    rec = recorder.arrays()
    outcome = recorder.outcome or "timeout"
    contact = recorder.contact
    label = classify(rec, outcome, contact)
    row = {
        "outcome": outcome,
        "label": label,
        "success": int(outcome == "success"),
        "collision": int(outcome == "collision"),
        "timeout": int(outcome == "timeout"),
        "freeze_any": int(has_freezing_event(rec)),
        "steps": steps,
        "nav_time": steps * C.DT if outcome == "success" else np.nan,
        "path_length": path_length if outcome == "success" else np.nan,
        "min_obs_dist": float(np.min(rec["clearance"])) if len(rec["clearance"]) else np.nan,
        "avg_velocity": float(np.mean(rec["v"])) if len(rec["v"]) else 0.0,
        "collision_kind": collision_kind(outcome, contact),
        "freeze_steps": int(freezing_condition(rec).sum()),
    }
    # sensitivity analysis over (delta, N)
    for df in C.SENS_DELTA_FRACS:
        for n in C.SENS_N:
            row[sens_key(df, n)] = int(has_freezing_event(rec, df * C.V_MAX * C.DT, n))
    return row
