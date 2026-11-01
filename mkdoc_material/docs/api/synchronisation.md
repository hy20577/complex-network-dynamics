# Synchronisation

Measures of how synchronised the nodes of a network are. Phases and time series are `(n_points, n_nodes)` arrays.

| Function | MATLAB name | Measures |
|----------|-------------|----------|
| `hilbert_phases` | `hilbert_phases` | Instantaneous phases \(\theta_i(t)\) from the Hilbert transform |
| `global_sync` | `global_sync` | Running average of the Kuramoto order parameter |
| `pairwise_sync` | `pairwise_sync_of_phases` | Phase locking between every pair of nodes |
| `global_order` | `globalorder_map` | Synchrony \(R(t)\) of Rubido et al. |
| `deviation_from_sync` | `deviation_from_sync` | Average distance between two oscillators over time |

## Definitions

**Kuramoto order parameter** (`global_sync` returns its running average):

\[
\rho(t) = \frac{1}{M}\Big|\sum_{j=1}^{M} e^{i\theta_j(t)}\Big|
\]

**Pairwise phase synchronisation** (Gómez-Gardeñes et al., 2010):

\[
r_{ij} = \Big|\big\langle e^{i(\theta_i(t)-\theta_j(t))}\big\rangle_t\Big|
\]

**Global order** (Rubido et al.): the variance over time of the network average \(\bar x(t)\), divided by the average over nodes of each node's variance over time:

\[
R(t) = \frac{\langle \bar x^2\rangle_t - \langle \bar x\rangle_t^2}{\frac{1}{M}\sum_i \big(\langle x_i^2\rangle_t - \langle x_i\rangle_t^2\big)}
\]

All three are 1 for fully synchronised nodes.

## Example

```python
import numpy as np
from complex_network_dynamics import load_network, stochastic_kuramoto, global_sync, pairwise_sync

A = load_network("16nodes")
rng = np.random.default_rng(0)
w = rng.uniform(-np.pi, np.pi, 16)
t, phases = stochastic_kuramoto(rng.random(16) * 2 * np.pi, A, w, K=10, D=1, dt=0.01, n_steps=20_000, rng=1)

global_sync(phases)[-1]   # overall synchrony
pairwise_sync(phases)     # 16 x 16 matrix
```

## API

::: complex_network_dynamics.synchronisation.hilbert_phases

::: complex_network_dynamics.synchronisation.global_sync

::: complex_network_dynamics.synchronisation.pairwise_sync

::: complex_network_dynamics.synchronisation.global_order

::: complex_network_dynamics.synchronisation.deviation_from_sync
