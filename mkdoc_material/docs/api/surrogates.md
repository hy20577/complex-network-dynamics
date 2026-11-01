# Surrogate Data

Surrogate datasets keep some properties of the data and destroy others. Comparing a measure such as MIR on the data with its values on many surrogates tells you whether the measured dependence is real.

All generators take time series as an `(n_points, n_nodes)` array and return `(n_surrogates, n_points, n_nodes)`. Pass `rng` (a seed or `numpy.random.Generator`) for reproducible results.

| Function | MATLAB name | Keeps | Destroys |
|----------|-------------|-------|----------|
| `amplitude_phase_surrogates` | `HT_mix_A_theta` | Range of amplitudes and phases | Amplitude correlation and phase synchronisation (fractions `pc_amplitude`, `pc_phase` of time points) |
| `phaseran` | `twnsd` | Power spectrum of each node and linear cross-correlations | Nonlinear structure |
| `random_surrogates` | `randsd` | Shape only | Everything |
| `correlated_random_surrogates` | `corr_randsd` | A fixed correlation between all pairs | Everything else |

## Amplitude-phase modulated surrogate data (APMSD)

Each node's analytic signal is computed with the Hilbert transform:

\[
z(t) = x(t) + i\,\mathcal{H}[x](t) = A(t)\, e^{i\theta(t)}
\]

In each surrogate, a random fraction `pc_amplitude` of the time points get amplitudes drawn uniformly from \([\min A, \max A]\), and a separate random fraction `pc_phase` get phases drawn uniformly from \([\min\theta, \max\theta]\). The surrogate is \(\mathrm{Re}\big[\tilde A(t)\, e^{i\tilde\theta(t)}\big]\).

## Network inference with surrogates

`mir_surrogate_test` (on the [Network Inference](network_inference.md) page) runs the whole method:

```python
from complex_network_dynamics import load_network, mir_surrogate_test

# traj: (n_points, n_nodes) time series, e.g. from simulate_discrete_models
mir_overall, adjacency, tp_fp = mir_surrogate_test(
    traj, p_sl=0.01, time_range=[1000, 2000, 5000],
    surrogate="amplitude_phase", pc_amplitude=1.0, pc_phase=0.5,
    true_network=load_network("16nodes"), rng=0,
)
adjacency[-1]   # inferred network from the longest data
tp_fp           # data length, true positive rate, false positive rate
```

!!! note "Not included"
    The MATLAB option `IAAFT2` uses a third-party GPL-licensed file and is not part of this library.

## API

::: complex_network_dynamics.surrogates.amplitude_phase_surrogates

::: complex_network_dynamics.surrogates.phaseran

::: complex_network_dynamics.surrogates.random_surrogates

::: complex_network_dynamics.surrogates.correlated_random_surrogates
