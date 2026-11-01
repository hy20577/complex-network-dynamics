"""Compare mutual_information, double_normalised_mir, lagged_double_mir and
best_case_tpr_fpr with the original MATLAB functions (run in Octave)."""
from pathlib import Path

import numpy as np
import pytest

from complex_network_dynamics import (
    best_case_tpr_fpr,
    double_normalised_mir,
    lagged_double_mir,
    mutual_information,
)

DATA = Path(__file__).parent / "reference" / "data"


def load(name):
    return np.loadtxt(DATA / name, delimiter=",", ndmin=2)


@pytest.fixture(scope="module")
def series():
    return load("in_mir_series.txt")


@pytest.mark.parametrize("grid, tag", [(5, "5"), ((4, 6), "46")])
def test_mutual_information(series, grid, tag):
    mi, p_null = mutual_information(series, grid)
    upper = np.triu_indices(series.shape[1], k=1)  # MATLAB fills only the upper triangle
    np.testing.assert_allclose(mi[upper], load(f"ref_mutual_info_{tag}_MI.txt")[upper], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(p_null[upper], load(f"ref_mutual_info_{tag}_Pnull.txt")[upper], rtol=1e-8, atol=1e-14)
    np.testing.assert_array_equal(mi, mi.T)


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
