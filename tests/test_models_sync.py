"""Synchronisation measures and the Kuramoto, Rossler, Hindmarsh-Rose and
composite logistic models, compared with the original MATLAB code (run in
Octave), plus checks against finite differences and simple properties."""
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert

from complex_network_dynamics import (
    composite_logistic,
    deviation_from_sync,
    df_composite_logistic,
    global_order,
    global_sync,
    hilbert_phases,
    hindmarsh_rose,
    hindmarsh_rose_jacobian,
    kuramoto,
    kuramoto_jacobian,
    pairwise_sync,
    rossler,
    rossler_jacobian,
    simulate_discrete_models,
    stochastic_kuramoto,
)

DATA = Path(__file__).parent / "reference" / "data"


def load(name):
    return np.loadtxt(DATA / name, delimiter=",", ndmin=2)


def finite_difference_jacobian(func, y, h=1e-6, **params):
    J = np.empty((len(y), len(y)))
    for k in range(len(y)):
        step = np.zeros(len(y))
        step[k] = h
        J[:, k] = (func(0.0, y + step, **params) - func(0.0, y - step, **params)) / (2 * h)
    return J


def split_extended(ref, n):
    """MATLAB extended output [f; J(:)] -> f, J (J stored column by column)."""
    ref = ref.ravel()
    return ref[:n], ref[n:].reshape(n, n, order="F")


# --------------------------------------------------------------------------- synchronisation


@pytest.fixture(scope="module")
def phases():
    return load("in_sync_phases.txt")


@pytest.fixture(scope="module")
def series():
    return load("in_mir_series.txt")


def test_global_sync(phases):
    np.testing.assert_allclose(global_sync(phases), load("ref_global_sync.txt").ravel(), rtol=1e-12)


def test_pairwise_sync(phases):
    np.testing.assert_allclose(pairwise_sync(phases), load("ref_pairwise_sync.txt"), rtol=1e-12, atol=1e-15)


def test_global_order(series):
    got = global_order(series)
    ref = load("ref_globalorder_map.txt").ravel()
    # The first point is the variance of a single sample, 0/0: NaN in Python,
    # while MATLAB's rounding can leave a tiny denominator and return 0
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], ref[1:], rtol=1e-9)


def test_deviation_from_sync(series):
    t = 0.1 * np.arange(1, len(series) + 1)
    ref = load("ref_deviation_from_sync.txt").ravel()
    got = deviation_from_sync(series[:, 0], series[:, 1], t)
    assert ref[-1] == 0  # the MATLAB output ends with an unfilled 0, left out in Python
    np.testing.assert_allclose(got, ref[:-1], rtol=1e-12)


def test_hilbert_phases_of_a_sine_grow_linearly():
    t = np.arange(4000) * 0.01
    phases = hilbert_phases(np.column_stack([np.sin(2 * t), np.cos(3 * t)]))
    np.testing.assert_array_equal(phases, np.unwrap(np.angle(hilbert(np.column_stack([np.sin(2 * t), np.cos(3 * t)]), axis=0)), axis=0))
    middle = slice(500, 3500)  # away from the edge effects of the transform
    slopes = np.polyfit(t[middle], phases[middle], 1)[0]
    np.testing.assert_allclose(slopes, [2, 3], rtol=1e-3)


def test_sync_measures_on_identical_and_independent_phases():
    rng = np.random.default_rng(0)
    same = np.repeat(rng.random((1000, 1)) * 6, 4, axis=1)
    np.testing.assert_allclose(global_sync(same), 1.0)
    np.testing.assert_allclose(pairwise_sync(same), 1.0)
    independent = rng.random((20000, 4)) * 2 * np.pi
    assert global_sync(independent)[-1] < 0.5
    off = ~np.eye(4, dtype=bool)
    assert pairwise_sync(independent)[off].max() < 0.05


# --------------------------------------------------------------------------- models against MATLAB


@pytest.fixture(scope="module")
def network():
    return load("in_logistic_adjacency.txt"), load("in_model_w.txt").ravel()


def test_kuramoto_matches_matlab(network):
    A, w = network
    theta = load("in_model_state_kuramoto.txt").ravel()
    f_ref, J_ref = split_extended(load("ref_kuramoto_ext.txt"), 12)
    np.testing.assert_allclose(kuramoto(0, theta, A, w, K=2.0), f_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(kuramoto_jacobian(0, theta, A, w, K=2.0), J_ref, rtol=1e-12, atol=1e-14)


def test_rossler_matches_matlab(network):
    A, w = network
    state = load("in_model_state_3M.txt").ravel()
    f_ref, J_ref = split_extended(load("ref_rossler_ext.txt"), 36)
    params = dict(A=A, K=0.3, w=w, a=0.15, f=0.2, c=10)
    np.testing.assert_allclose(rossler(0, state, **params), f_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(rossler_jacobian(0, state, **params), J_ref, rtol=1e-12, atol=1e-14)


def test_hindmarsh_rose_matches_matlab(network):
    A, _ = network
    state = load("in_model_state_3M.txt").ravel()
    f_ref, J_ref = split_extended(load("ref_hr_ext.txt"), 36)
    np.testing.assert_allclose(hindmarsh_rose(0, state, A, g=0.2), f_ref, rtol=1e-12, atol=1e-13)

    # The Jacobian matches MATLAB except dp_i/dp_i, where HR_ext.m has
    # -3p^2 + 3p but the derivative of +3p^2 is 6p (checked below)
    J = hindmarsh_rose_jacobian(0, state, A, g=0.2)
    p = state[:12]
    diag = np.arange(12)
    expected = J_ref.copy()
    expected[diag, diag] += 3 * p
    np.testing.assert_allclose(J, expected, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("f, df, params", [
    (kuramoto, kuramoto_jacobian, dict(K=2.0)),
    (rossler, rossler_jacobian, dict(K=0.3, a=0.15, f=0.2, c=10)),
    (hindmarsh_rose, hindmarsh_rose_jacobian, dict(g=0.2)),
])
def test_jacobians_match_finite_differences(network, f, df, params):
    A, w = network
    if f is kuramoto:
        params = dict(params, w=w)
        state = load("in_model_state_kuramoto.txt").ravel()
    else:
        state = load("in_model_state_3M.txt").ravel()
    np.testing.assert_allclose(df(0, state, A, **params), finite_difference_jacobian(f, state, A=A, **params), atol=1e-6)


# --------------------------------------------------------------------------- composite logistic


def test_composite_logistic_matches_matlab(network):
    A, _ = network
    x0 = load("in_logistic_x0.txt").ravel()
    # Composite_log_generator.m: 5 transient + 5 recorded steps, 10 Jacobians
    traj, lyap = simulate_discrete_models(
        x0, composite_logistic, df_composite_logistic, A, t=11, alpha=0.1, n_transient=0, r=4.0, n=3
    )
    np.testing.assert_allclose(traj[5:10], load("ref_composite_orbits.txt"), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(lyap, load("ref_composite_lyapunov.txt").ravel(), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("n", [1, 2, 4])
def test_composite_logistic_derivative(n):
    x = np.linspace(0.05, 0.95, 7)
    h = 1e-7
    numeric = (composite_logistic(x + h, n=n) - composite_logistic(x - h, n=n)) / (2 * h)
    np.testing.assert_allclose(df_composite_logistic(x, n=n), numeric, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- stochastic Kuramoto


def test_stochastic_kuramoto_without_noise_is_euler(network):
    A, w = network
    theta0 = load("in_model_state_kuramoto.txt").ravel()
    _, phases = stochastic_kuramoto(theta0, A, w, K=2.0, D=0.0, dt=0.01, n_steps=50, rng=0)
    theta = theta0.copy()
    for k in range(50):
        theta = theta + 0.01 * kuramoto(0, theta, A, w, K=2.0)
    np.testing.assert_allclose(phases[-1], theta, rtol=1e-12)


def test_stochastic_kuramoto_noise_has_the_right_size():
    # Uncoupled oscillators: increments are w dt + D sqrt(dt) N(0, 1)
    M = 3
    w = np.array([0.5, -1.0, 2.0])
    t, phases = stochastic_kuramoto(np.zeros(M), np.zeros((M, M)), w, K=0.0, D=2.0, dt=0.1, n_steps=200_000, rng=1)
    increments = np.diff(phases, axis=0)
    np.testing.assert_allclose(increments.mean(axis=0), w * 0.1, atol=0.005)
    np.testing.assert_allclose(increments.std(axis=0), 2.0 * np.sqrt(0.1), rtol=0.01)
    assert len(t) == 200_001
