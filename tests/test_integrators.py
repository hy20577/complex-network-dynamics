"""The integrators with Lyapunov spectrum: the known Lorenz spectrum, agreement
between the plain and compiled RK4, shared time and output conventions, and
input checks."""
import numpy as np
import pytest

from complex_network_dynamics import (
    build_rk4_lyapunov,
    euler_with_lyapunov,
    lorenz,
    lorenz_jacobian,
    lorenz_jacobian_njit,
    lorenz_njit,
    rk4_with_lyapunov,
    solve_ivp_with_lyapunov,
)

ARGS = (10.0, 28.0, 8.0 / 3.0)
LORENZ_SPECTRUM = np.array([0.906, 0.0, -14.572])  # Sprott (2003)


@pytest.fixture(scope="module")
def lorenz_solver():
    return build_rk4_lyapunov(lorenz_njit, lorenz_jacobian_njit)


@pytest.fixture(scope="module")
def y_on_attractor():
    _, traj = rk4_with_lyapunov(lorenz, np.ones(3), 0.0, 0.01, 2, lyap_mode="none", n_transient=2000)
    return traj[-1]


def test_rk4_gives_the_lorenz_spectrum(y_on_attractor):
    _, _, spectrum = rk4_with_lyapunov(lorenz, y_on_attractor, 0.0, 0.01, 100_001, df=lorenz_jacobian)
    np.testing.assert_allclose(spectrum, LORENZ_SPECTRUM, atol=0.03)


def test_solve_ivp_gives_the_lorenz_spectrum(y_on_attractor):
    _, _, spectrum = solve_ivp_with_lyapunov(lorenz, y_on_attractor, 0.0, 500.0, df=lorenz_jacobian, dt_qr=0.1)
    np.testing.assert_allclose(spectrum, LORENZ_SPECTRUM, atol=0.05)


@pytest.mark.parametrize("lyap_mode", ["none", "spectrum", "history"])
def test_compiled_rk4_matches_plain_rk4(lorenz_solver, y_on_attractor, lyap_mode):
    """Same method, same names, same outputs: equal to rounding error, even for short runs."""
    kwargs = dict(lyap_mode=lyap_mode, n_transient=50, qr_interval=7)
    plain = rk4_with_lyapunov(lorenz, y_on_attractor, 2.0, 0.01, 200, df=lorenz_jacobian, **kwargs)
    compiled = lorenz_solver(y_on_attractor, 2.0, 0.01, 200, ARGS, **kwargs)
    assert len(plain) == len(compiled)
    for a, b in zip(plain, compiled):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("run", ["euler", "rk4", "compiled", "solve_ivp"])
def test_time_runs_on_through_the_transient(lorenz_solver, run):
    t0, h, n_transient = 3.0, 0.01, 40
    if run == "euler":
        t, _ = euler_with_lyapunov(lorenz, np.ones(3), t0, h, 11, lyap_mode="none", n_transient=n_transient)
    elif run == "rk4":
        t, _ = rk4_with_lyapunov(lorenz, np.ones(3), t0, h, 11, lyap_mode="none", n_transient=n_transient)
    elif run == "compiled":
        t, _ = lorenz_solver(np.ones(3), t0, h, 11, ARGS, lyap_mode="none", n_transient=n_transient)
    else:
        t, _ = solve_ivp_with_lyapunov(lorenz, np.ones(3), t0, 10 * h, lyap_mode="none", n_transient=n_transient, dt_qr=h)
    np.testing.assert_allclose(t, t0 + h * (n_transient + np.arange(11)))


def test_solve_ivp_reaches_the_requested_time():
    # int(0.7 / 0.1) is 6 in floating point; the run must still reach 0.7
    t, _ = solve_ivp_with_lyapunov(lorenz, np.ones(3), 0.0, 0.7, lyap_mode="none", n_transient=0, dt_qr=0.1)
    assert len(t) == 8
    np.testing.assert_allclose(t[-1], 0.7)


@pytest.mark.parametrize("run", ["euler", "rk4", "compiled", "solve_ivp"])
def test_history_ends_at_the_spectrum(lorenz_solver, run):
    common = dict(n_transient=10)
    if run == "euler":
        call = lambda mode: euler_with_lyapunov(lorenz, np.ones(3), 0.0, 0.001, 101, df=lorenz_jacobian, lyap_mode=mode, **common)
    elif run == "rk4":
        call = lambda mode: rk4_with_lyapunov(lorenz, np.ones(3), 0.0, 0.01, 101, df=lorenz_jacobian, lyap_mode=mode, **common)
    elif run == "compiled":
        call = lambda mode: lorenz_solver(np.ones(3), 0.0, 0.01, 101, ARGS, lyap_mode=mode, **common)
    else:
        call = lambda mode: solve_ivp_with_lyapunov(lorenz, np.ones(3), 0.0, 1.0, df=lorenz_jacobian, lyap_mode=mode, dt_qr=0.1, **common)
    _, _, spectrum = call("spectrum")
    t, _, lyap_times, lyap_history = call("history")
    np.testing.assert_allclose(np.sort(lyap_history[-1])[::-1], spectrum)
    assert lyap_times[-1] <= t[-1] and np.all(np.diff(lyap_times) > 0)


@pytest.mark.parametrize("run", ["euler", "rk4", "compiled", "solve_ivp"])
def test_invalid_options_are_rejected(lorenz_solver, run):
    calls = {
        "euler": lambda **kw: euler_with_lyapunov(lorenz, np.ones(3), 0.0, 0.01, 50, **kw),
        "rk4": lambda **kw: rk4_with_lyapunov(lorenz, np.ones(3), 0.0, 0.01, 50, **kw),
        "solve_ivp": lambda **kw: solve_ivp_with_lyapunov(lorenz, np.ones(3), 0.0, 1.0, **kw),
    }
    if run == "compiled":
        with pytest.raises(ValueError, match="Invalid lyap_mode"):
            lorenz_solver(np.ones(3), 0.0, 0.01, 50, ARGS, lyap_mode="spectra")
        return
    with pytest.raises(ValueError, match="Invalid lyap_mode"):
        calls[run](df=lorenz_jacobian, lyap_mode="spectra")
    with pytest.raises(ValueError, match="must be provided"):
        calls[run](lyap_mode="spectrum")
