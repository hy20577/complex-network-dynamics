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
    # The MATLAB code divided the running sum by the time t_i = 0.1 i instead of by i,
    # so its values are 1/0.1 times ours; its first entry is i = 2.
    ref = load("ref_deviation_from_sync.txt").ravel()
    got = deviation_from_sync(series[:, 0], series[:, 1])
    assert ref[-1] == 0  # the MATLAB output ends with an unfilled 0
    np.testing.assert_allclose(got[1:len(ref)], 0.1 * ref[:-1], rtol=1e-12)


def test_deviation_from_sync_is_the_running_average_distance():
    x1, x2 = np.zeros(500), np.ones(500)
    np.testing.assert_allclose(deviation_from_sync(x1, x2), 1.0)
    with pytest.raises(ValueError, match="same length"):
        deviation_from_sync(np.zeros(3), np.zeros(4))


def test_hilbert_phases_of_a_sine_grow_linearly():
    t = np.arange(4000) * 0.01
    phases = hilbert_phases(np.column_stack([np.sin(2 * t), np.cos(3 * t)]), demean=False)
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
    (hindmarsh_rose, hindmarsh_rose_jacobian, dict(g=0.2, a=1.2, b=2.5, c=0.8, d=4.0, r=0.01, s=3.5, p0=-1.4, I_ext=3.0)),
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


def test_kuramoto_follows_the_row_convention():
    """dtheta_i/dt = w_i + (K/M) sum_j A_ij sin(theta_j - theta_i): row i drives node i."""
    rng = np.random.default_rng(3)
    M, K = 6, 3.0
    A = (rng.random((M, M)) < 0.4).astype(float)
    np.fill_diagonal(A, 0.0)
    w, theta = rng.uniform(-np.pi, np.pi, M), rng.uniform(0, 2 * np.pi, M)
    expected = w + K / M * np.array([A[i] @ np.sin(theta - theta[i]) for i in range(M)])
    np.testing.assert_allclose(kuramoto(0, theta, A, w, K), expected)


@pytest.mark.parametrize("func", [kuramoto, kuramoto_jacobian])
def test_kuramoto_rejects_inputs_of_different_sizes(func):
    A, w = np.ones((4, 4)), np.zeros(4)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.zeros(5), A, w)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.zeros(4), A, np.zeros(3))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        stochastic_kuramoto(np.zeros(5), A, w, n_steps=1)


@pytest.mark.parametrize("func", [rossler, rossler_jacobian])
def test_rossler_rejects_inputs_of_different_sizes(func):
    A = np.ones((4, 4))
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.zeros(11), A)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.zeros(12), A, w=np.ones(3))
    func(0, np.zeros(12), A, w=np.ones(4))  # one frequency per node is fine


def test_hindmarsh_rose_coupling_is_diffusive():
    """With C = D - A, the coupling term is +g sum_l A_il (p_l - p_i), on a directed network too."""
    rng = np.random.default_rng(4)
    M, g = 6, 0.3
    A = (rng.random((M, M)) < 0.4).astype(float)
    np.fill_diagonal(A, 0.0)
    state = rng.normal(size=3 * M)
    uncoupled = hindmarsh_rose(0, state, A, g=0.0)[:M]
    p = state[:M]
    expected = uncoupled + g * np.array([A[i] @ (p - p[i]) for i in range(M)])
    np.testing.assert_allclose(hindmarsh_rose(0, state, A, g=g)[:M], expected)


@pytest.mark.parametrize("func", [hindmarsh_rose, hindmarsh_rose_jacobian])
def test_hindmarsh_rose_rejects_a_state_of_the_wrong_size(func):
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.zeros(11), np.ones((4, 4)))


def test_demeaning_removes_the_hilbert_offset_artefact():
    """Two independent signals on [0, 1] look phase-locked unless the mean is removed."""
    rng = np.random.default_rng(5)
    x = np.empty((20_000, 2))
    x[0] = rng.random(2)
    for k in range(1, len(x)):
        x[k] = 4 * x[k - 1] * (1 - x[k - 1])
    assert pairwise_sync(hilbert_phases(x, demean=False))[0, 1] > 0.3
    assert pairwise_sync(hilbert_phases(x))[0, 1] < 0.05
    shifted = np.sin(0.01 * np.arange(5000))[:, None] + 3.0
    np.testing.assert_allclose(hilbert_phases(shifted), hilbert_phases(shifted - 3.0))
