"""Regression guards for the shared Experiment C metrics.

These synthetic cases pin the metric definitions without needing private
per-sample fidelity archives or pretrained checkpoints.
"""
import numpy as np

from experiments.validation.fidelity_metrics import HUG, STATS, agg, class_table, per_class_samples


def _sample_data():
    return {
        "cls": np.array([1, 0, 1, 0, 1], dtype=int),
        "len_ratio": np.array([2., 1., 4., 3., 6.]),
        "rough": np.array([.2, .1, .4, .3, .6]),
        "minclear": np.array([.015, .014, .019, .016, .010]),
        "div": np.array([.25, .5, np.nan]),
    }


def test_class_table_preserves_original_statistics_and_missing_classes():
    a = class_table(_sample_data(), ncls=3)
    assert a.shape == (3, 5)
    np.testing.assert_allclose(a[0], [2., .2, .015, .5, .25])
    np.testing.assert_allclose(a[1], [4., .4, (.015 + .019 + .010) / 3, 1 / 3, .5])
    assert np.all(np.isnan(a[2]))
    assert HUG == .015


def test_grouped_samples_preserve_original_order():
    out = per_class_samples(_sample_data(), ncls=3, key="len_ratio")
    assert out == [[1., 3.], [2., 4., 6.], []]


def test_aggregate_uses_ratio_vs_difference_and_respects_layout_weights():
    r = np.array([[1., 2., 4., .1, 5.], [3., 4., 8., .2, 10.], [np.nan] * 5])
    g = np.array([[2., 6., 2., .3, 10.], [5., 8., 4., .5, 20.], [np.nan] * 5])
    lid = np.array([0, 1, 1]); wcls = np.array([1., 1., 0.])
    stats = agg(g, r, lid, wcls, np.array([1., 2.]))
    assert STATS == ["len_ratio", "rough", "clear", "hug", "div"]
    # weighted means: layout 0 has weight 1, layout 1 has weight 2.
    np.testing.assert_allclose(stats["len_ratio"], (2 + 2 * 5) / 3 - (1 + 2 * 3) / 3)
    np.testing.assert_allclose(stats["rough"], (6 + 2 * 8) / (2 + 2 * 4))
    np.testing.assert_allclose(stats["clear"], (2 + 2 * 4) / (4 + 2 * 8))
    np.testing.assert_allclose(stats["hug"], (.3 + 2 * .5) / 3 - (.1 + 2 * .2) / 3)
    np.testing.assert_allclose(stats["div"], (10 + 2 * 20) / (5 + 2 * 10))
