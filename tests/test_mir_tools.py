"""Compare double_normalised_mir, lagged_double_mir and best_case_tpr_fpr with
the original MATLAB functions (run in Octave)."""
from pathlib import Path

import numpy as np
import pytest

from complex_network_dynamics import (
    best_case_tpr_fpr,
    double_normalised_mir,
    lagged_double_mir,
)

DATA = Path(__file__).parent / "reference" / "data"


def load(name):
    return np.loadtxt(DATA / name, delimiter=",", ndmin=2)


@pytest.fixture(scope="module")
def series():
    return load("in_mir_series.txt")


def test_double_normalised_mir(series):
    normalised, max_rate = double_normalised_mir(series, load("in_best_case_true.txt"))
    np.testing.assert_allclose(normalised, load("ref_double_normalised_mir.txt"), rtol=1e-10, atol=1e-14)
    assert max_rate == pytest.approx(load("ref_double_normalised_max_rate.txt").item(), rel=1e-12)


def test_double_normalised_mir_without_network(series):
    normalised, max_rate = double_normalised_mir(series)
    assert max_rate is None
    assert normalised.max() == pytest.approx(1.0)


def test_lagged_double_mir(series):
    lagged_mi, lagged_mir = lagged_double_mir(series[:500], 2)
    np.testing.assert_allclose(lagged_mi, load("ref_lagged_mi.txt"), rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(lagged_mir, load("ref_lagged_mir.txt"), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize(
    "case, similarity",
    [(1, "in_best_case_similarity.txt"), (2, "ref_mir_auto_Mir.txt")],  # case 2 is upper-triangular
)
def test_best_case_tpr_fpr(case, similarity):
    roc, best, inferred = best_case_tpr_fpr(load(similarity), load("in_best_case_true.txt"))
    np.testing.assert_allclose(roc, load(f"ref_best_case_roc_case{case}.txt"), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(best, load(f"ref_best_case_best_case{case}.txt"), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(inferred, inferred.T)
    assert set(np.unique(inferred[np.triu_indices(len(inferred), k=1)])) <= {0.0, 1.0}


# --------------------------------------------------------------------------- input checks


def test_scaled_mir_needs_three_nodes(series):
    for bad in (series[:, :2], series[:, 0]):
        with pytest.raises(ValueError, match="at least three nodes"):
            double_normalised_mir(bad)
        with pytest.raises(ValueError, match="at least three nodes"):
            lagged_double_mir(bad, 1)


def test_double_normalised_mir_rejects_a_constant_node(series):
    data = series.copy()
    data[:, 2] = 0.3
    with pytest.raises(ValueError, match=r"Node\(s\) \[2\]"):
        double_normalised_mir(data)


def test_double_normalised_mir_checks_the_true_network(series):
    true = load("in_best_case_true.txt")
    with pytest.raises(ValueError, match="binary"):
        double_normalised_mir(series, 0.5 * true)
    with pytest.raises(ValueError, match="shape"):
        double_normalised_mir(series, true[:-1, :-1])
    with pytest.raises(ValueError, match="no links"):
        double_normalised_mir(series, np.zeros_like(true))


@pytest.mark.parametrize("max_lag", [-1, 1.5, 500])
def test_lagged_double_mir_rejects_a_bad_lag(series, max_lag):
    with pytest.raises(ValueError, match="max_lag"):
        lagged_double_mir(series[:500], max_lag)


def test_lagged_double_mir_without_lag_is_symmetric_and_scaled(series):
    lagged_mi, lagged_mir = lagged_double_mir(series[:500], 0)
    np.testing.assert_array_equal(lagged_mi, lagged_mi.T)
    assert lagged_mir.max() == pytest.approx(1.0)
    assert np.all(lagged_double_mir(series[:500], 2)[0] >= lagged_mi)  # more lags can only raise the MI


def test_best_case_tpr_fpr_finds_a_perfect_threshold():
    true = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    similarity = np.array([[0, 0.9, 0.1], [0.9, 0, 0.8], [0.1, 0.8, 0]])
    roc, best, inferred = best_case_tpr_fpr(similarity, true)
    assert roc.shape == (3, 4)
    np.testing.assert_allclose(best[0, :3], [1, 0, 0])
    np.testing.assert_array_equal(inferred, true)


def test_best_case_tpr_fpr_checks_its_inputs():
    true = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    with pytest.raises(ValueError, match="square"):
        best_case_tpr_fpr(np.ones((3, 2)), true)
    with pytest.raises(ValueError, match="different shapes"):
        best_case_tpr_fpr(np.ones((4, 4)), true)
    with pytest.raises(ValueError, match="binary"):
        best_case_tpr_fpr(np.ones((3, 3)), 2 * true)
