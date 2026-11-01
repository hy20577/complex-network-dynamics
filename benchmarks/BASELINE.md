# Speed baseline

Recorded on 2026-11-01 before any optimisation, with `python benchmarks/run_benchmarks.py`
(Python 3.12.7, NumPy 1.26.4, Apple Silicon Mac). Best of 3 runs (1 run for the slow cases).

| Case                                                    |  Time (s) |
|---------------------------------------------------------|-----------|
| simulate_discrete_models: 16 nodes, 10k steps           |     0.192 |
| Grid: 16 nodes, 5k points                               |     0.035 |
| corr_decay: 16 nodes, 5k points, N=10                   |     0.247 |
| MIR: 16 nodes, 5k points, N=3..10                       |     1.056 |
| rk4_with_lyapunov: Lorenz, 50k steps                    |     1.360 |
| rk4_with_lyapunov: coupled Lorenz 12 nodes, 5k steps    |     0.658 |
| solve_ivp_with_lyapunov: Lorenz, t=100                  |     0.808 |
| build_rk4_lyapunov (Numba): Lorenz, 50k steps           |     0.138 |
