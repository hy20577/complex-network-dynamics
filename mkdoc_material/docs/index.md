# Complex Network Dynamics

Python toolkit for complex networks of nonlinear oscillators: simulate maps and ODE networks, compute Lyapunov spectra and synchronisation, and infer network structure from time series.

## Features

<div class="grid cards" markdown>

-   **[Discrete Models](api/discrete_models.md)**

    Logistic and circle maps, and networks of coupled maps with the full Lyapunov spectrum.

-   **[Continuous Models](api/continuous_models.md)**

    Lorenz, Rössler, Hindmarsh–Rose, Kuramoto (also stochastic) and Lotka–Volterra on networks, with Jacobians.

-   **[Integrators](api/integrators.md)**

    Euler, RK4, SciPy adaptive and compiled RK4 solvers that compute Lyapunov exponents.

-   **[Network Inference](api/network_inference.md)**

    Reconstruct networks from time series with the Mutual Information Rate.

-   **[Surrogate Data](api/surrogates.md)**

    Amplitude-phase modulated surrogates (APMSD) and a statistical test that decides which links are real.

-   **[Synchronisation](api/synchronisation.md)**

    Hilbert phases, the Kuramoto order parameter and pairwise phase locking.

-   **[Network Metrics](api/network_metrics.md)**

    Diameter and average path length for connected and disconnected graphs.

-   **[Datasets](api/datasets.md)**

    Example adjacency matrices that ship with the package.

</div>

## Installation

Requires Python 3.10–3.13. From the project root:

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `numba`, `networkx`.

## Quick start

```python
import numpy as np
from complex_network_dynamics import (
    load_network, logistic, df_logistic, simulate_discrete_models, MIR, plot_similarity_gap_bars,
)

# 1. Load a network
A = load_network("16nodes")

# 2. Simulate coupled logistic maps on it
traj, lyap = simulate_discrete_models(np.random.rand(16), logistic, df_logistic, A, t=10_000, alpha=0.1)

# 3. Infer the network back from the time series
mir, mi = MIR(traj)
plot_similarity_gap_bars(mir, A)
```

## All functions

| Module | Functions |
|--------|-----------|
| [`datasets`](api/datasets.md) | `load_network` |
| [`discrete_models`](api/discrete_models.md) | `logistic`, `df_logistic`, `circle`, `df_circle`, `composite_logistic`, `df_composite_logistic`, `simulate_discrete_models` |
| [`continuous_models`](api/continuous_models.md) | `lorenz`, `lorenz_jacobian`, `coupled_lorenz`, `coupled_lorenz_jacobian`, `lotka_volterra`, `lotka_volterra_jacobian`, `kuramoto`, `kuramoto_jacobian`, `stochastic_kuramoto`, `rossler`, `rossler_jacobian`, `hindmarsh_rose`, `hindmarsh_rose_jacobian`, `lorenz_njit`, `lorenz_jacobian_njit`, `coupled_lorenz_njit`, `coupled_lorenz_jacobian_njit` |
| [`integrators`](api/integrators.md) | `euler_with_lyapunov`, `rk4_with_lyapunov`, `solve_ivp_with_lyapunov`, `rk4_lyapunov_fast`, `build_rk4_lyapunov`, `_apply_qr_update` |
| [`network_inference`](api/network_inference.md) | `MIR`, `mir_surrogate_test`, `fdr_bh`, `tpr_fpr`, `double_normalised_mir`, `lagged_double_mir`, `best_case_tpr_fpr`, `plot_similarity_gap_bars`, `Grid`, `corr_decay`, `compute_single_pair_mi`, `location_jit` |
| [`synchronisation`](api/synchronisation.md) | `hilbert_phases`, `global_sync`, `pairwise_sync`, `global_order`, `deviation_from_sync` |
| [`surrogates`](api/surrogates.md) | `amplitude_phase_surrogates`, `phaseran`, `random_surrogates`, `correlated_random_surrogates` |
| [`network_metrics`](api/network_metrics.md) | `calculate_distance_metrics` |

## Citing

This library implements the methods from Yıldırım & Antonopoulos, *Chaos, Solitons & Fractals* 188, 115554 (2024) and the accompanying PhD thesis. See [Publications](publications.md) for links to the article and the thesis, and BibTeX entries.
