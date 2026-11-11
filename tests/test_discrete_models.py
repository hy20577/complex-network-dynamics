"""The logistic and circle maps, their derivatives, and the coupled maps of
Eq. (13) of the APMSD paper with their Lyapunov spectrum."""
import numpy as np
import pytest

from complex_network_dynamics import circle, df_circle, df_logistic, load_network, logistic, simulate_discrete_models


def test_circle_derivative_matches_finite_differences():
    # Points where x + r - (K / 2 pi) sin(2 pi x) does not cross an integer within +-h
    x, h = np.array([0.05, 0.2, 0.33, 0.61, 0.9]), 1e-7
    numerical = (circle(x + h) - circle(x - h)) / (2 * h)
    np.testing.assert_allclose(df_circle(x), numerical, atol=1e-5)


def test_circle_stays_in_the_unit_interval():
    x = np.random.default_rng(0).random(10_000)
    for _ in range(50):
        x = circle(x)
        assert np.all((x >= 0) & (x < 1))


def test_uncoupled_logistic_maps_have_exponent_ln2():
    A = load_network("16nodes")
    x0 = np.random.default_rng(1).random(len(A))
    _, lyap = simulate_discrete_models(x0, logistic, df_logistic, A, 20_000, alpha=0.0)
    np.testing.assert_allclose(lyap, np.log(2), atol=0.02)


def test_coupling_follows_eq_13():
    """x_i(n+1) = (1 - alpha) f(x_i) + (alpha / k_i) sum_j A_ij f(x_j), on a directed network."""
    rng = np.random.default_rng(2)
    M, alpha = 6, 0.2
    A = (rng.random((M, M)) < 0.5).astype(float)
    np.fill_diagonal(A, 0.0)
    A[0] = 0.0  # an isolated node evolves as (1 - alpha) f(x_0)
    traj, _ = simulate_discrete_models(rng.random(M), circle, df_circle, A, 2, alpha=alpha, n_transient=0)
    fx = circle(traj[0])
    k = A.sum(axis=1)
    coupled = np.divide(A @ fx, k, out=np.zeros(M), where=k > 0)
    np.testing.assert_allclose(traj[1], (1 - alpha) * fx + alpha * coupled)


def test_simulate_discrete_models_rejects_a_state_of_the_wrong_size():
    with pytest.raises(ValueError, match="Dimension mismatch"):
        simulate_discrete_models(np.ones(5), logistic, df_logistic, np.ones((4, 4)), 10)
