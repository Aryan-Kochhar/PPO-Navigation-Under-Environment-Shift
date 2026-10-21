"""Sanity tests for the simulator and the pre-registered failure taxonomy.

Run:  python -m pytest -q      (or)      python tests/test_metrics.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navsim import config as C                       # noqa: E402
from navsim.env import NavEnv                        # noqa: E402
from navsim.metrics import (EpisodeRecorder, classify, has_corridor_deadlock,   # noqa: E402
                            has_freezing_event, has_oscillation_near_goal,
                            is_collision_from_behind)


def _empty_scene(env, rects=()):
    """Put the robot at (2,5) facing a goal at (8,5) with no moving obstacles."""
    env.reset(seed=0)
    env.rects = np.array(rects, dtype=np.float64).reshape(-1, 4)
    env.ob_pos = np.zeros((0, 2))
    env.ob_head = np.zeros(0)
    env.ob_spd = np.zeros(0)
    env.ob_phase = np.zeros(0, dtype=int)
    env.pos = np.array([2.0, 5.0])
    env.goal = np.array([8.0, 5.0])
    env.theta = 0.0
    env.d_goal = 6.0


def _stand_still(env, n):
    rec = EpisodeRecorder()
    for _ in range(n):
        _, _, _, _, info = env.step(np.array([-1.0, 0.0]))   # v = 0, w = 0
        rec.record(env, info)
    return rec.arrays()


def test_freezing_in_open_space():
    """Stopped with a clear path to the goal -> freezing."""
    env = NavEnv(diagnostics=True)
    _empty_scene(env)
    rec = _stand_still(env, C.FREEZE_N + 5)
    assert rec["a_free"].min() >= 1
    assert has_freezing_event(rec)


def test_yielding_is_not_freezing():
    """Stopped in front of a wall with no collision-free advancing action -> NOT freezing."""
    env = NavEnv(diagnostics=True)
    _empty_scene(env, rects=[[2.3, 0.5, 2.6, 9.5]])
    rec = _stand_still(env, C.FREEZE_N + 5)
    assert rec["a_free"].max() == 0
    assert not has_freezing_event(rec)


def test_short_hesitation_is_not_freezing():
    env = NavEnv(diagnostics=True)
    _empty_scene(env)
    rec = _stand_still(env, C.FREEZE_N - 5)
    assert not has_freezing_event(rec)


def _synthetic(T, **over):
    rec = {"progress": np.full(T, 0.03), "a_free": np.full(T, 5.0), "omega": np.zeros(T),
           "v": np.full(T, 0.3), "dist_goal": np.linspace(5, 0.5, T), "x": np.linspace(0, 5, T),
           "y": np.zeros(T), "clear_l": np.full(T, 2.0), "clear_r": np.full(T, 2.0),
           "clearance": np.full(T, 1.0)}
    rec.update(over)
    return rec


def test_oscillation_near_goal():
    T = 60
    w = np.zeros(T)
    w[30:50] = np.where(np.arange(20) % 2 == 0, 0.8, -0.8)
    dist = np.full(T, 3.0)
    dist[25:] = 0.6
    rec = _synthetic(T, omega=w, dist_goal=dist)
    assert has_oscillation_near_goal(rec)
    rec_far = _synthetic(T, omega=w, dist_goal=np.full(T, 3.0))
    assert not has_oscillation_near_goal(rec_far)


def test_corridor_deadlock():
    T = 40
    rec = _synthetic(T, clear_l=np.full(T, 0.1), clear_r=np.full(T, 0.1), a_free=np.zeros(T),
                     x=np.full(T, 3.0), progress=np.zeros(T))
    assert has_corridor_deadlock(rec)
    assert not has_freezing_event(rec)          # mutually exclusive by construction


def test_collision_from_behind():
    assert is_collision_from_behind({"kind": "dynamic", "bearing": math.radians(170), "closing_along_heading": False})
    assert is_collision_from_behind({"kind": "dynamic", "bearing": math.radians(100), "closing_along_heading": True})
    assert not is_collision_from_behind({"kind": "dynamic", "bearing": 0.1, "closing_along_heading": True})
    assert not is_collision_from_behind({"kind": "static", "bearing": np.nan, "closing_along_heading": False})


def test_taxonomy_order_and_residual():
    T = 40
    frozen = _synthetic(T, progress=np.zeros(T))
    behind = {"kind": "dynamic", "bearing": math.pi, "closing_along_heading": True}
    front = {"kind": "dynamic", "bearing": 0.0, "closing_along_heading": False}
    assert classify(frozen, "collision", behind) == "collision_from_behind"   # test 1 wins
    assert classify(frozen, "collision", front) == "freezing"                # then test 2
    assert classify(_synthetic(T), "timeout", None) == "timeout"
    assert classify(_synthetic(T), "success", None) == "success"


def test_frontal_static_collision():
    """Amendment (test 6): head-on and wall/block contacts get their own label, but
    only after the original five tests, so it never steals their episodes."""
    T = 40
    front = {"kind": "dynamic", "bearing": 0.0, "closing_along_heading": False}
    static = {"kind": "static", "bearing": np.nan, "closing_along_heading": False}
    wall = {"kind": "wall", "bearing": np.nan, "closing_along_heading": False}
    for contact in (front, static, wall):
        assert classify(_synthetic(T), "collision", contact) == "frontal_static_collision"
    # froze first, then hit a wall -> still freezing (test 2 outranks test 6)
    assert classify(_synthetic(T, progress=np.zeros(T)), "collision", wall) == "freezing"
    # a contact from behind is still test 1
    behind = {"kind": "dynamic", "bearing": math.pi, "closing_along_heading": True}
    assert classify(_synthetic(T), "collision", behind) == "collision_from_behind"


def test_all_cells_run():
    for cell in C.ALL_CELLS:
        env = NavEnv(cell=cell, diagnostics=True)
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        for _ in range(30):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            assert np.all(np.abs(obs) <= 1.0 + 1e-6) and np.isfinite(r)
            if term or trunc:
                break


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
