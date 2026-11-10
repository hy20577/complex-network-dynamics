"""The single and coupled Lorenz systems: Jacobians against finite differences,
the Numba versions against the plain ones, and simple properties of the
coupling (Eq. (20) of the APMSD paper)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from complex_network_dynamics import (
    coupled_lorenz,
    coupled_lorenz_jacobian,
    coupled_lorenz_jacobian_njit,
    coupled_lorenz_njit,
    lorenz,
    lorenz_jacobian,
    lorenz_jacobian_njit,
    lorenz_njit,
)

PARAMS = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)


def finite_difference_jacobian(func, y, h=1e-6, **params):
    J = np.empty((len(y), len(y)))
    for k in range(len(y)):
        step = np.zeros(len(y))
        step[k] = h
        J[:, k] = (func(0.0, y + step, **params) - func(0.0, y - step, **params)) / (2 * h)
    return J


@pytest.fixture(scope="module")
def networks():
    """A symmetric, a directed and a weighted network of 6 nodes."""
    rng = np.random.default_rng(1)
    n = 6
    upper = np.triu((rng.random((n, n)) < 0.4).astype(float), 1)
    symmetric = upper + upper.T
    directed = (rng.random((n, n)) < 0.4).astype(float)
    np.fill_diagonal(directed, 0.0)
    weighted = rng.random((n, n)) * symmetric
    return {"symmetric": symmetric, "directed": directed, "weighted": weighted}


@pytest.fixture(scope="module")
def state():
    rng = np.random.default_rng(2)
    y = 5 * rng.normal(size=18)
    y[12:] += 20
    return y


# --------------------------------------------------------------------------- single system


def test_lorenz_returns_a_flat_vector_that_scipy_accepts():
    assert lorenz(0.0, [1.0, 1.0, 1.0]).shape == (3,)
    assert solve_ivp(lorenz, (0, 1), [1.0, 1.0, 1.0]).success


def test_lorenz_jacobian_matches_finite_differences():
    y = np.array([1.3, -0.7, 20.0])
    np.testing.assert_allclose(lorenz_jacobian(0, y, **PARAMS), finite_difference_jacobian(lorenz, y, **PARAMS), atol=1e-6)


def test_lorenz_njit_matches_plain():
    y = np.array([1.3, -0.7, 20.0])
    args = tuple(PARAMS.values())
    np.testing.assert_array_equal(lorenz_njit(0.0, y, args), lorenz(0.0, y, **PARAMS))
    np.testing.assert_array_equal(lorenz_jacobian_njit(0.0, y, args), lorenz_jacobian(0.0, y, **PARAMS))


# --------------------------------------------------------------------------- coupled system


@pytest.mark.parametrize("kind", ["symmetric", "directed", "weighted"])
def test_coupled_lorenz_jacobian_matches_finite_differences(networks, state, kind):
    A = networks[kind]
    np.testing.assert_allclose(
        coupled_lorenz_jacobian(0, state, A, K=0.7, **PARAMS),
        finite_difference_jacobian(coupled_lorenz, state, A=A, K=0.7, **PARAMS),
        atol=1e-6,
    )


@pytest.mark.parametrize("kind", ["symmetric", "directed", "weighted"])
def test_coupled_lorenz_njit_matches_plain(networks, state, kind):
    A = networks[kind]
    args = (A, 0.7, *PARAMS.values())
    np.testing.assert_array_equal(coupled_lorenz_njit(0.0, state, args), coupled_lorenz(0.0, state, A, K=0.7, **PARAMS))
    np.testing.assert_array_equal(
        coupled_lorenz_jacobian_njit(0.0, state, args), coupled_lorenz_jacobian(0.0, state, A, K=0.7, **PARAMS)
    )


def test_coupled_lorenz_follows_the_row_convention(networks, state):
    """dx_i/dt = sigma (y_i - x_i) + K sum_j A_ij (x_j - x_i): row i drives node i."""
    A, n, K = networks["directed"], 6, 0.7
    x, y = state[:n], state[n : 2 * n]
    expected = PARAMS["sigma"] * (y - x) + K * np.array([A[i] @ (x - x[i]) for i in range(n)])
    np.testing.assert_allclose(coupled_lorenz(0, state, A, K=K, **PARAMS)[:n], expected)


def test_uncoupled_nodes_are_single_lorenz_systems(networks, state):
    n = 6
    f = coupled_lorenz(0, state, networks["symmetric"], K=0.0, **PARAMS)
    for i in range(n):
        idx = [i, n + i, 2 * n + i]
        np.testing.assert_allclose(f[idx], lorenz(0, state[idx], **PARAMS))


def test_strong_coupling_synchronises_two_nodes():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    y0 = [1.0, -3.0, 2.0, 4.0, 25.0, 20.0]
    coupled = solve_ivp(coupled_lorenz, (0, 200), y0, args=(A, 5.0), rtol=1e-8, atol=1e-10)
    free = solve_ivp(coupled_lorenz, (0, 200), y0, args=(A, 0.0), rtol=1e-8, atol=1e-10)
    assert abs(coupled.y[0, -1] - coupled.y[1, -1]) < 1e-6
    assert abs(free.y[0, -1] - free.y[1, -1]) > 1e-2


@pytest.mark.parametrize("func", [coupled_lorenz, coupled_lorenz_jacobian])
def test_coupled_lorenz_rejects_a_state_of_the_wrong_size(networks, func):
    with pytest.raises(ValueError, match="Dimension mismatch"):
        func(0, np.ones(5), networks["symmetric"])
