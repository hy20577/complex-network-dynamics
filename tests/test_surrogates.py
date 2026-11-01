"""Tests for the surrogate data method (APMSD) and its statistics.

fdr_bh, tpr_fpr and the full mir_surrogate_test pipeline are compared exactly
with the original MATLAB code (run in Octave, see tests/reference/README.md).
Surrogates are random, so the generators are checked by their properties.
"""
from pathlib import Path

import numpy as np
import pytest

from complex_network_dynamics import (
    amplitude_phase_surrogates,
    correlated_random_surrogates,
    fdr_bh,
    mir_surrogate_test,
    phaseran,
    random_surrogates,
    tpr_fpr,
)

DATA = Path(__file__).parent / "reference" / "data"


def load(name):
    return np.loadtxt(DATA / name, delimiter=",", ndmin=2)


@pytest.fixture(scope="module")
def series():
    return load("in_mir_series.txt")[:, :5]


# --------------------------------------------------------------------------- against MATLAB


@pytest.mark.parametrize("method", ["pdep", "dep"])
@pytest.mark.parametrize("case", [1, 2, 3])
def test_fdr_bh(method, case):
    pvals = load("in_fdr_pvals.txt")[case - 1]
    ref = load(f"ref_fdr_{method}_case{case}.txt")
    h, crit_p, adj_ci_cvrg, adj_p = fdr_bh(pvals, 0.05, method)
    np.testing.assert_array_equal(h, ref[0].astype(bool))
    np.testing.assert_allclose(adj_p, ref[1], rtol=1e-12)
    assert crit_p == pytest.approx(ref[2, 0], rel=1e-12)
    if np.isnan(ref[3, 0]):
        assert np.isnan(adj_ci_cvrg)
    else:
        assert adj_ci_cvrg == pytest.approx(ref[3, 0], rel=1e-12)


def test_tpr_fpr():
    tpr, fpr = tpr_fpr(load("in_tpr_true.txt"), load("in_tpr_inferred.txt"))
    np.testing.assert_allclose([tpr, fpr], load("ref_tpr_fpr.txt").ravel(), rtol=1e-12)


def test_mir_surrogate_test_matches_matlab(series):
    # The same stored surrogates as the Octave run, handed out in the same order
    stored = load("in_surrogate_fixed.txt").reshape(20, 1000, 5)
    calls = iter(range(20))
    fixed = lambda data, rng: stored[next(calls), : len(data)]

    mir_overall, adjacency, tp_fp = mir_surrogate_test(
        series, p_sl=0.1, time_range=[500, 1000], surrogate=fixed,
        true_network=load("in_surrogate_true_network.txt"),
    )
    for it in range(2):
        np.testing.assert_allclose(mir_overall[it], load(f"ref_surrogate_mir_overall_len{it + 1}.txt"), rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(adjacency[it], load(f"ref_surrogate_adjacency_len{it + 1}.txt"))
    np.testing.assert_allclose(tp_fp, load("ref_surrogate_tp_fp.txt"), rtol=1e-12)


# --------------------------------------------------------------------------- surrogate properties


def test_phaseran_keeps_spectrum_and_correlations(series):
    surr = phaseran(series, n_surrogates=3, rng=0)
    data = series[:999]  # 1000 points: one is dropped to make the length odd
    assert surr.shape == (3, 999, 5)
    for s in surr:
        np.testing.assert_allclose(np.abs(np.fft.fft(s, axis=0)), np.abs(np.fft.fft(data, axis=0)), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.corrcoef(s.T), np.corrcoef(data.T), atol=1e-10)
        assert not np.allclose(s, data)


def test_amplitude_phase_with_nothing_randomised_returns_data(series):
    # Real part of the analytic signal is the signal itself
    surr = amplitude_phase_surrogates(series, n_surrogates=2, pc_amplitude=0, pc_phase=0, rng=0)
    for s in surr:
        np.testing.assert_allclose(s, series, atol=1e-12)


@pytest.mark.parametrize("pc_amplitude, pc_phase", [(0.3, 0), (0, 0.3), (1, 0.5)])
def test_amplitude_phase_changes_the_right_number_of_points(series, pc_amplitude, pc_phase):
    surr = amplitude_phase_surrogates(series, n_surrogates=4, pc_amplitude=pc_amplitude, pc_phase=pc_phase, rng=1)
    n = len(series)
    for s in surr:
        changed = np.any(~np.isclose(s, series, atol=1e-12), axis=1).sum()
        if pc_amplitude == 0 or pc_phase == 0:
            assert changed == int(np.ceil(max(pc_amplitude, pc_phase) * n))
        else:
            assert changed >= int(np.ceil(pc_amplitude * n))


def test_amplitude_phase_values_stay_within_amplitude_range(series):
    from scipy.signal import hilbert

    amp_max = np.abs(hilbert(series, axis=0)).max()
    surr = amplitude_phase_surrogates(series, n_surrogates=3, rng=2)
    assert np.all(np.abs(surr) <= amp_max + 1e-12)


def test_surrogates_are_reproducible_with_a_seed(series):
    for make in (phaseran, amplitude_phase_surrogates, random_surrogates, correlated_random_surrogates):
        np.testing.assert_array_equal(make(series, 2, rng=5), make(series, 2, rng=5))


def test_correlated_random_surrogates_have_the_requested_correlation(series):
    surr = correlated_random_surrogates(series, n_surrogates=1, correlation=0.5, n_points=200_000, rng=3)[0]
    corr = np.corrcoef(surr.T)
    off = ~np.eye(5, dtype=bool)
    np.testing.assert_allclose(corr[off], 0.5, atol=0.01)
    assert surr.shape == (200_000, 5)


def test_random_surrogates_shape_and_range(series):
    surr = random_surrogates(series, n_surrogates=2, rng=4)
    assert surr.shape == (2,) + series.shape
    assert surr.min() >= 0 and surr.max() < 1
