"""Tests for the independent checkers (experiments/validation/indep_geom.py) and chart edge cases (Experiment A1).

  python -m pytest tests/test_chart_soundness.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "validation"))
import indep_geom as G  # noqa: E402
from topoflow import chart  # noqa: E402
from topoflow.env import Box, Circle, Env  # noqa: E402

rng = np.random.default_rng(0)


def brute_dist(P, Q, o, k=20001):
    t = np.linspace(0, 1, k)[:, None]
    S = P * (1 - t) + Q * t
    if o.kind == "circle":
        return max(float(np.min(np.hypot(S[:, 0] - o.cx, S[:, 1] - o.cy) - o.r)), 0.0)
    qx, qy = np.abs(S[:, 0] - o.cx) - o.hw, np.abs(S[:, 1] - o.cy) - o.hh
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    return max(float(d.min()), 0.0)


@pytest.mark.parametrize("kind", ["circle", "box"])
def test_seg_dist_matches_bruteforce_and_exact(kind):
    for _ in range(300):
        o = Circle(*rng.uniform(-0.5, 0.5, 2), rng.uniform(0.05, 0.3)) if kind == "circle" else \
            Box(*rng.uniform(-0.5, 0.5, 2), *rng.uniform(0.05, 0.3, 2))
        P, Q = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
        d = float(G.seg_obstacle_dist(P[None], Q[None], o)[0])
        b = brute_dist(P, Q, o)
        assert d <= b + 1e-12            # analytic distance is a lower bound of any sampled distance
        assert b - d < 1e-4              # and the sampled distance converges to it
        c = 0.02
        ok, _ = G.exact_clearance_ok(P, Q, o, c)
        assert ok == (d >= c) or abs(d - c) < 1e-12


def test_box_corner_case():
    o = Box(0.0, 0.0, 0.1, 0.1)
    P, Q = np.array([0.0, 0.3]), np.array([0.3, 0.0])   # passes the corner (0.1, 0.1) at distance 0.2/sqrt2 - ...
    d = float(G.seg_obstacle_dist(P[None], Q[None], o)[0])
    assert abs(d - (0.3 / np.sqrt(2) - 0.1 * np.sqrt(2))) < 1e-12
    # c = 0.06: the square-inflated rectangle (corner region) is met, the Euclidean distance is still > c
    assert d > 0.06
    assert bool(G.seg_meets_rect(*P, *Q, -0.16, 0.16, -0.16, 0.16, open_=True)) is True
    # c = 0.05: the line x + y = 0.3 only touches the corner of the open rectangle -> no interior meet
    assert bool(G.seg_meets_rect(*P, *Q, -0.15, 0.15, -0.15, 0.15, open_=True)) is False


def test_crossing_word_basic():
    o = [Circle(0.0, 0.0, 0.2)]
    below = np.array([[-0.9, 0.0], [0.0, -0.5], [0.9, 0.0]])
    above = np.array([[-0.9, 0.0], [0.0, 0.5], [0.9, 0.0]])
    assert G.crossing_word(below, o) == ((0, 1),)
    assert G.crossing_word(above, o) == ()
    # below to the right, come back left below, then go over the top: crossings cancel
    loop = np.array([[-0.9, 0.0], [0.3, -0.5], [-0.3, -0.5], [-0.3, 0.5], [0.9, 0.0]])
    assert G.crossing_word(loop, o) == ()
    # vertex exactly on the cut is counted once
    on = np.array([[-0.9, 0.0], [-0.5, -0.5], [0.0, -0.5], [0.5, -0.5], [0.9, 0.0]])
    assert G.crossing_word(on, o) == ((0, 1),)


def _check_chart(env, code, c, U):
    lo, hi, ok = chart.bounds(env, code, c, 60)
    if not ok:
        return None
    P = chart.to_path(env, chart.decode(U, lo, hi), c)
    worst = np.inf
    for path in P:
        for o in env.obstacles:
            worst = min(worst, float(G.seg_obstacle_dist(path[:-1], path[1:], o).min()) - c)
        assert G.crossing_word(path, env.obstacles) == G.word_of_code(code)
    return worst


def test_saturated_and_infinite_latents():
    env = Env([Circle(-0.3, 0.05, 0.2), Box(0.25, -0.05, 0.12, 0.18)])
    U = np.stack([np.full(60, v) for v in (50.0, -50.0, 1e6, -1e6, np.inf, -np.inf)] +
                 [rng.choice([-np.inf, np.inf], 60) for _ in range(4)])
    for code in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w = _check_chart(env, code, 0.01, U)
        assert w is None or w >= 0


def test_narrow_throat_and_degenerate_band():
    # two obstacles leaving a gap of 2c + gap0 between them at the throat (different sides)
    c = 0.01
    for gap0 in (1e-3, 1e-6, 1e-9):
        r = 0.2
        env = Env([Circle(-0.1, 0.0, r), Circle(0.1 + 2 * r * 0, 2 * r + 2 * c + gap0, r)])
        for code in ((1, 0), (0, 1), (0, 0), (1, 1)):
            w = _check_chart(env, code, c, rng.normal(0, 3, (16, 60)))
            assert w is None or w >= 0


def test_obstacle_inside_endpoint_range():
    # A3 violated: inflated x-range of the first obstacle contains start x. Soundness must still hold
    # (bands include the obstacle in the first segment) or the chart is reported infeasible.
    env = Env([Circle(-0.85, 0.3, 0.1), Circle(0.2, 0.0, 0.2)])
    for code in ((0, 0), (0, 1), (1, 0), (1, 1)):
        lo, hi, ok = chart.bounds(env, code, 0.01, 60)
        if ok:
            P = chart.to_path(env, chart.decode(rng.normal(0, 3, (16, 60)), lo, hi), 0.01)
            for path in P:
                for o in env.obstacles:
                    assert float(G.seg_obstacle_dist(path[:-1], path[1:], o).min()) >= 0.01
