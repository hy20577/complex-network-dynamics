# Speed benchmarks

Measured with `python benchmarks/run_benchmarks.py` on an Apple Silicon Mac (10 cores),
Python 3.12.7, NumPy 1.26.4. Best of 3 runs (1 run for the slow cases).
Numba compile time is not counted.

| Case                                                    | Before (s) | After (s) | Speed-up |
|---------------------------------------------------------|-----------:|----------:|---------:|
| Grid: 16 nodes, 5k points                               |      0.041 |     0.005 |     8×   |
| corr_decay: 16 nodes, 5k points, N=10                   |      0.251 |     0.006 |    42×   |
| MIR: 16 nodes, 5k points, N=3..10                       |      0.856 |     0.025 |    34×   |
| MIR: 8 nodes, 4k points, N=20..30                       |      9.291 |     0.206 |    45×   |
| mir_surrogate_test: 16 nodes, 5k points, 100 surrogates |     37.825 |   3–4     |   ~10×   |
| simulate_discrete_models: 16 nodes, 10k steps           |      0.190 |     0.194 |        – |
| rk4_with_lyapunov: Lorenz, 50k steps                    |      1.415 |     1.481 |        – |
| rk4_with_lyapunov: coupled Lorenz 12 nodes, 5k steps    |      0.670 |     0.808 |        – |
| solve_ivp_with_lyapunov: Lorenz, t=100                  |      0.814 |     0.807 |        – |
| build_rk4_lyapunov (Numba): Lorenz, 50k steps           |      0.163 |     0.153 |        – |

"Before" is commit `f063334`; "After" is the MIR optimisation that followed it.
Outputs of MIR, corr_decay, Grid and mir_surrogate_test are bit-for-bit identical
before and after.

## What changed

- MI and the correlation decay time are computed together for each pair, so
  each pair is binned once per grid size, and all pairs run in parallel with Numba.
- The decay time uses a breadth-first search on the transition graph instead of
  building a SciPy sparse matrix and calling a general shortest-path routine.
  The graph is stored as a dense table for grids up to 16 × 16 and as a sorted
  edge list for larger ones, where the table would grow as N⁴.
- `Grid` counts occupied cells directly, for all pairs in parallel.

The remaining time in `mir_surrogate_test` is about 0.025 s per MIR call
(101 calls) plus about 0.5 s of surrogate generation. The integrators have not
been optimised yet.
