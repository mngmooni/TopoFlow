import numpy as np
import pytest

from topoflow.env import Box, Circle, Env, e0_env, random_layout
from topoflow.routes import make_dataset, resample_arclength, sample_route
from topoflow.topology import (crossing_word, expected_word_for_code, reduce_word, str_to_word,
                               winding_vector, word_to_str)


# ---------------------------------------------------------------- word algebra
def test_reduce_word_cancels_adjacent_inverses():
    assert reduce_word([1, -1]) == ()
    assert reduce_word([1, 2, -2, -1, 3]) == (3,)
    assert reduce_word([1, 1, -1]) == (1,)
    assert reduce_word([1, -2, 2, 2]) == (1, 2)


def test_commutator_is_nontrivial_but_has_zero_winding():
    w = reduce_word(str_to_word("abAB"))
    assert w == str_to_word("abAB")           # nothing cancels
    assert winding_vector(w, 2).tolist() == [0, 0]  # abelianization can't see it


def test_word_str_roundtrip():
    for s in ["e", "a", "aB", "abAB", "ccA"]:
        assert word_to_str(str_to_word(s)) == s


# ---------------------------------------------------------------- E0 paths
def poly(*pts):
    return np.array(pts, dtype=float)


def test_e0_above_and_below():
    env = e0_env()
    above = poly((-0.9, 0), (0, 0.5), (0.9, 0))
    below = poly((-0.9, 0), (0, -0.5), (0.9, 0))
    assert crossing_word(env, above) == ()
    assert crossing_word(env, below) == (1,)
    assert not env.path_collides(above) and not env.path_collides(below)


def test_e0_full_loop_gives_a_squared():
    env = e0_env()
    # below (a) -> up the right side -> back left above -> down the left side -> below again (a) -> goal
    loop = poly((-0.9, 0), (-0.4, -0.5), (0.4, -0.5), (0.5, 0.5), (-0.5, 0.5),
                (-0.5, -0.6), (0.6, -0.6), (0.9, 0))
    assert not env.path_collides(loop)
    assert crossing_word(env, loop) == (1, 1)


def test_e0_backtrack_cancels():
    env = e0_env()
    # go under the cut rightwards (a), come back leftwards under it (A), then go above -> e
    back = poly((-0.9, 0), (-0.3, -0.5), (0.2, -0.5), (-0.3, -0.6), (-0.5, 0.0),
                (0.0, 0.5), (0.9, 0))
    assert not env.path_collides(back)
    assert crossing_word(env, back) == ()


def test_straight_line_collides_in_e0():
    env = e0_env()
    assert env.path_collides(poly((-0.9, 0), (0.9, 0)))


def test_resampling_does_not_create_silent_collision():
    """Resampling can cut corners; every dataset path is checked after resampling."""
    env = e0_env()
    rng = np.random.default_rng(0)
    ds = make_dataset(env, 200, rng, T=16)
    assert not env.collision_mask(ds.X).any()


# ---------------------------------------------------------------- boxes
def test_box_collision_and_word():
    env = Env([Box(0.0, 0.0, 0.2, 0.3)])
    env.validate()
    assert env.path_collides(poly((-0.9, 0), (0.9, 0)))
    assert crossing_word(env, poly((-0.9, 0), (0, -0.5), (0.9, 0))) == (1,)
    assert crossing_word(env, poly((-0.9, 0), (0, 0.5), (0.9, 0))) == ()


# ---------------------------------------------------------------- cut validity
def test_validate_rejects_cut_hitting_other_obstacle():
    env = Env([Circle(0.0, 0.3, 0.1), Circle(0.05, -0.3, 0.15)])
    with pytest.raises(ValueError):
        env.validate()


def test_validate_rejects_start_on_cut():
    env = Env([Circle(0.0, 0.3, 0.1)], start=(0.0, -0.5), goal=(0.9, 0.0))
    with pytest.raises(ValueError):
        env.validate()


# ---------------------------------------------------------------- code vs word
@pytest.mark.parametrize("seed", range(5))
def test_code_matches_word_on_random_layouts(seed):
    rng = np.random.default_rng(seed)
    env = random_layout(rng, n_obs=3)
    ds = make_dataset(env, 120, rng, T=32)
    assert not env.collision_mask(ds.X).any()
    for code, w in zip(ds.code, ds.word):
        assert w == expected_word_for_code(tuple(code))
    for s in ds.stats["per_class"].values():
        assert s["mismatch"] == 0


def test_endpoints_fixed():
    env = e0_env()
    rng = np.random.default_rng(1)
    p = sample_route(env, (1,), rng, T=32)
    assert np.allclose(p[0], env.start) and np.allclose(p[-1], env.goal)
    q = resample_arclength(p, 10)
    assert np.allclose(q[0], env.start) and np.allclose(q[-1], env.goal)


def test_imbalance_is_respected():
    env = e0_env()
    rng = np.random.default_rng(2)
    ds = make_dataset(env, 2000, rng, codes=[(0,), (1,)], class_probs={(0,): 0.9, (1,): 0.1})
    frac_below = ds.code[:, 0].mean()
    assert 0.08 < frac_below < 0.12
