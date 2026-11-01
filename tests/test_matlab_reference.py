"""Compare the Python ports with the original MATLAB thesis code.

Reference outputs in tests/reference/data were produced by running the
unmodified MATLAB functions in Octave (see tests/reference/README.md).
"""
from pathlib import Path

import numpy as np
import pytest

from complex_network_dynamics import (
    MIR,
    Grid,
    corr_decay,
    df_logistic,
    euler_with_lyapunov,
    location_jit,
    logistic,
    lorenz,
    lorenz_jacobian,
    rk4_with_lyapunov,
    simulate_discrete_models,
)

DATA = Path(__file__).parent / "reference" / "data"


def load(name):
    return np.loadtxt(DATA / name, delimiter=",", ndmin=2)


@pytest.fixture(scope="module")
def series():
    return load("in_mir_series.txt")


# --------------------------------------------------------------------------- location


@pytest.mark.parametrize("N", [4, 7])
def test_location_random_points(N):
    X = load("in_location_random.txt")
    coords, ppb = location_jit(X, N)
    ref_coords = load(f"ref_location_random_N{N}_coords.txt")
    ref_ppb = load(f"ref_location_random_N{N}_ppb.txt")

    # Python: 0-based point index and flat cell index; MATLAB: 1-based
    np.testing.assert_array_equal(coords[:, 1:3], ref_coords[:, 1:3])
    np.testing.assert_array_equal(coords[:, 3] + 1, ref_coords[:, 3])
    np.testing.assert_array_equal(ppb[:, 1], ref_ppb[:, 1])


@pytest.mark.parametrize("N", [4, 7])
def test_location_points_on_grid_lines(N):
    X = load("in_location_gridlines.txt")
    coords, _ = location_jit(X, N)
    ref_coords = load(f"ref_location_gridlines_N{N}_coords.txt")
    np.testing.assert_array_equal(coords[:, 1:3], ref_coords[:, 1:3])


# --------------------------------------------------------------------------- Grid, corr_decay


def test_grid(series):
    assert tuple(Grid(series)) == tuple(load("ref_grid.txt").ravel().astype(int))


@pytest.mark.parametrize("which", [0, 1])
def test_corr_decay_off_diagonal(series, which):
    N = int(load("ref_grid.txt").ravel()[which])
    ref = load(f"ref_corr_decay_N{N}.txt")
    got = corr_decay(series, N)
    off = ~np.eye(len(ref), dtype=bool)
    np.testing.assert_array_equal(got[off], ref[off])


# --------------------------------------------------------------------------- MIR


@pytest.mark.parametrize("case, N_range", [("auto", None), ("3_6", (3, 6))])
def test_mutual_information(series, case, N_range):
    _, mi = MIR(series, N_range=N_range)
    ref = load(f"ref_mir_{case}_Mi.txt")
    upper = np.triu_indices(len(ref))  # MATLAB fills only j >= i
    np.testing.assert_allclose(mi[upper], ref[upper], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case, N_range", [("auto", None), ("3_6", (3, 6))])
def test_mir_off_diagonal(series, case, N_range):
    mir, _ = MIR(series, N_range=N_range)
    ref = load(f"ref_mir_{case}_Mir.txt")
    upper = np.triu_indices(len(ref), k=1)
    np.testing.assert_allclose(mir[upper], ref[upper], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case, N_range", [("auto", None), ("3_6", (3, 6))])
def test_mir_diagonal(series, case, N_range):
    mir, _ = MIR(series, N_range=N_range)
    ref = load(f"ref_mir_{case}_Mir.txt")
    np.testing.assert_allclose(np.diag(mir), np.diag(ref), rtol=1e-10)


# --------------------------------------------------------------------------- coupled logistic maps

LOGISTIC_TRANSIENT = 10
LOGISTIC_STEPS = 3000


@pytest.fixture(scope="module")
def logistic_run():
    A = load("in_logistic_adjacency.txt")
    x0 = load("in_logistic_x0.txt").ravel()
    return simulate_discrete_models(
        x0, logistic, df_logistic, A, t=LOGISTIC_STEPS, alpha=0.1, n_transient=LOGISTIC_TRANSIENT, r=4.0
    )


def test_logistic_orbits(logistic_run):
    traj, _ = logistic_run
    ref = load("ref_logistic_orbits.txt")
    # r = 4 doubles rounding differences every step, so only the start is comparable
    np.testing.assert_allclose(traj[:25], ref[:25], rtol=0, atol=1e-8)


def test_logistic_lyapunov_exact_short_run():
    # 25 steps from x0 with no transient: the same Jacobians as Log_generator.m
    # with time_window = 10 and niter = 15, before the trajectories separate
    A = load("in_logistic_adjacency.txt")
    x0 = load("in_logistic_x0.txt").ravel()
    _, lyap = simulate_discrete_models(x0, logistic, df_logistic, A, t=26, alpha=0.1, n_transient=0, r=4.0)
    ref = load("ref_logistic_lyapunov_25_steps.txt").ravel()
    np.testing.assert_allclose(lyap, ref, rtol=1e-8, atol=1e-10)


def test_logistic_lyapunov_long_run(logistic_run):
    # After ~50 steps the two chaotic trajectories differ, so each 3000-step
    # average is a separate finite-time estimate of the same spectrum
    _, lyap = logistic_run
    ref = load("ref_logistic_lyapunov_final.txt").ravel()
    np.testing.assert_allclose(lyap, ref, rtol=0, atol=0.04)


def test_logistic_isolated_node():
    # Node 3 has no links: MATLAB evolves it as (1 - alpha) * f(x_3)
    A = load("in_logistic_isolated_adjacency.txt")
    x0 = load("in_logistic_isolated_x0.txt").ravel()
    # 12 states = 11 Jacobians, as in Log_generator.m (1 transient + 10 recorded steps)
    traj, lyap = simulate_discrete_models(x0, logistic, df_logistic, A, t=12, alpha=0.3, n_transient=0, r=4.0)
    ref_orbits = load("ref_logistic_isolated_orbits.txt")
    # Log_generator.m with time_window = 1 starts recording after one step
    np.testing.assert_allclose(traj[1:11], ref_orbits, rtol=1e-10, atol=1e-12)

    # The exponents are checked against the math, not against Log_generator.m:
    # its Gram-Schmidt skips a projection whenever a vector has a zero entry
    # (`if a ~= 0` on a vector), which happens for disconnected networks.
    # See tests/reference/README.md.
    states = traj[:11]
    isolated = np.mean(np.log(np.abs((1 - 0.3) * df_logistic(states[:, 2], r=4.0))))
    pair = np.array([[0.7, 0.3], [0.3, 0.7]])
    pair_sum = np.mean([np.log(abs(np.linalg.det(pair * df_logistic(x[:2], r=4.0)))) for x in states])
    assert np.isclose(lyap, isolated, rtol=1e-10).sum() == 1
    others = lyap[~np.isclose(lyap, isolated, rtol=1e-10)]
    np.testing.assert_allclose(others.sum(), pair_sum, rtol=1e-10)


# --------------------------------------------------------------------------- Lorenz: RK4 and Euler
# RK4.m / Euler.m store the state after each step and do no transient, so the
# Python run gets one extra step and its first (initial) row is skipped.


def test_rk4_lorenz():
    ref_orbits = load("ref_rk4_lorenz_orbits.txt")
    ref_lyap = load("ref_rk4_lorenz_lyapunov.txt").ravel()
    _, traj, spectrum = rk4_with_lyapunov(
        lorenz, load("in_lorenz_x0.txt").ravel(), 0.0, 0.01, len(ref_orbits) + 1,
        df=lorenz_jacobian, n_transient=0, qr_interval=1,
    )
    np.testing.assert_allclose(traj[1:], ref_orbits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(spectrum, ref_lyap, rtol=1e-4, atol=1e-6)


def test_euler_lorenz():
    ref_orbits = load("ref_euler_lorenz_orbits.txt")
    ref_lyap = load("ref_euler_lorenz_lyapunov.txt").ravel()
    _, traj, spectrum = euler_with_lyapunov(
        lorenz, load("in_lorenz_x0.txt").ravel(), 0.0, 0.001, len(ref_orbits) + 1,
        df=lorenz_jacobian, n_transient=0, qr_interval=1,
    )
    np.testing.assert_allclose(traj[1:], ref_orbits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(spectrum, ref_lyap, rtol=1e-4, atol=1e-6)
