"""The Lotka-Volterra predator-prey model: Jacobian against finite differences,
the equilibrium and its eigenvalues, and the conserved quantity."""
import numpy as np
from scipy.integrate import solve_ivp

from complex_network_dynamics import lotka_volterra, lotka_volterra_jacobian

PARAMS = dict(alpha=1.1, beta=0.4, delta=0.1, gamma=0.4)
EQUILIBRIUM = np.array([PARAMS["gamma"] / PARAMS["delta"], PARAMS["alpha"] / PARAMS["beta"]])


def test_jacobian_matches_finite_differences():
    y, h = np.array([3.0, 1.5]), 1e-6
    numerical = np.column_stack(
        [(lotka_volterra(0, y + h * e, **PARAMS) - lotka_volterra(0, y - h * e, **PARAMS)) / (2 * h) for e in np.eye(2)]
    )
    np.testing.assert_allclose(lotka_volterra_jacobian(0, y, **PARAMS), numerical, atol=1e-8)


def test_equilibrium_is_a_centre():
    """At (gamma/delta, alpha/beta) the flow stops and the eigenvalues are +-i sqrt(alpha gamma)."""
    np.testing.assert_allclose(lotka_volterra(0, EQUILIBRIUM, **PARAMS), 0.0, atol=1e-12)
    eigenvalues = np.linalg.eigvals(lotka_volterra_jacobian(0, EQUILIBRIUM, **PARAMS))
    np.testing.assert_allclose(eigenvalues.real, 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sort(eigenvalues.imag), np.array([-1.0, 1.0]) * np.sqrt(PARAMS["alpha"] * PARAMS["gamma"]))


def test_conserved_quantity_stays_constant():
    a, b, d, g = PARAMS["alpha"], PARAMS["beta"], PARAMS["delta"], PARAMS["gamma"]
    sol = solve_ivp(lotka_volterra, (0, 100), [10.0, 10.0], args=(a, b, d, g), rtol=1e-10, atol=1e-12)
    x, p = sol.y
    V = d * x - g * np.log(x) + b * p - a * np.log(p)
    assert np.ptp(V) < 1e-7
