"""Time the main library functions so speed-ups can be measured.

Run from the project root:  python benchmarks/run_benchmarks.py
Each case reports the best of a few runs. Numba functions are called once
first so compile time is not counted.
"""
import platform
import time

import numpy as np

from complex_network_dynamics import (
    MIR,
    Grid,
    build_rk4_lyapunov,
    corr_decay,
    coupled_lorenz,
    coupled_lorenz_jacobian,
    df_logistic,
    load_network,
    logistic,
    lorenz,
    lorenz_jacobian,
    lorenz_jacobian_njit,
    lorenz_njit,
    mir_surrogate_test,
    rk4_with_lyapunov,
    simulate_discrete_models,
    solve_ivp_with_lyapunov,
)


def best_time(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    rng = np.random.default_rng(0)
    A16 = load_network("16nodes")
    A12 = load_network("12nodes")
    traj16, _ = simulate_discrete_models(rng.random(16), logistic, df_logistic, A16, t=5000, alpha=0.1, r=4.0)
    noisy8 = rng.standard_normal((4000, 8))
    MIR(noisy8[:200, :3], N_range=(3, 4))  # compile
    mir_surrogate_test(traj16[:300, :4], p_sl=0.5, rng=0)  # compile
    x0_lorenz = np.ones(3)
    solver = build_rk4_lyapunov(lorenz_njit, lorenz_jacobian_njit)
    solver(x0_lorenz, 0.0, 0.01, 100, 10, (10.0, 28.0, 8 / 3))  # compile

    cases = [
        ("simulate_discrete_models: 16 nodes, 10k steps",
         lambda: simulate_discrete_models(rng.random(16), logistic, df_logistic, A16, t=10_000, alpha=0.1, r=4.0), 3),
        ("Grid: 16 nodes, 5k points", lambda: Grid(traj16), 3),
        ("corr_decay: 16 nodes, 5k points, N=10", lambda: corr_decay(traj16, 10), 3),
        ("MIR: 16 nodes, 5k points, N=3..10", lambda: MIR(traj16, N_range=(3, 10)), 1),
        ("MIR: 8 nodes, 4k points, N=20..30", lambda: MIR(noisy8, N_range=(20, 30)), 1),
        ("mir_surrogate_test: 16 nodes, 5k points, 100 surrogates",
         lambda: mir_surrogate_test(traj16, p_sl=0.01, rng=0), 1),
        ("rk4_with_lyapunov: Lorenz, 50k steps",
         lambda: rk4_with_lyapunov(lorenz, x0_lorenz, 0.0, 0.01, 50_000, df=lorenz_jacobian), 1),
        ("rk4_with_lyapunov: coupled Lorenz 12 nodes, 5k steps",
         lambda: rk4_with_lyapunov(coupled_lorenz, rng.random(36), 0.0, 0.01, 5_000,
                                   df=coupled_lorenz_jacobian, n_transient=0, A=A12, K=0.5), 1),
        ("solve_ivp_with_lyapunov: Lorenz, t=100",
         lambda: solve_ivp_with_lyapunov(lorenz, lorenz_jacobian, x0_lorenz, 100.0), 1),
        ("build_rk4_lyapunov (Numba): Lorenz, 50k steps",
         lambda: solver(x0_lorenz, 0.0, 0.01, 50_000, 10, (10.0, 28.0, 8 / 3)), 3),
    ]

    print(f"Python {platform.python_version()}, numpy {np.__version__}, {platform.machine()} {platform.system()}\n")
    print(f"| {'Case':<55} | {'Time (s)':>9} |")
    print(f"|{'-' * 57}|{'-' * 11}|")
    for name, fn, repeats in cases:
        print(f"| {name:<55} | {best_time(fn, repeats):>9.3f} |", flush=True)


if __name__ == "__main__":
    main()
