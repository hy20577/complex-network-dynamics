# Network Inference

Reconstruct a network from the time series of its nodes using the **Mutual Information Rate (MIR)**.

## How it works

For every pair of nodes \((i, j)\):

1. The joint trajectory \((x_i, x_j)\) is normalized and binned on an \(N \times N\) grid (`location_jit`).
2. The mutual information \(I_{ij}\) is computed from the grid occupancy (`compute_single_pair_mi`):
   \[
   I_{ij} = \sum_{a,b} p_{ab} \ln \frac{p_{ab}}{p_a\, p_b}
   \]
3. The correlation decay time \(T_{ij}\) is the largest shortest path in the graph of cell-to-cell transitions (`corr_decay`).
4. MIR is \(I_{ij} / T_{ij}\), averaged over a range of grid sizes \(N\) chosen by `Grid`.

Pairs with high MIR are likely to be directly connected.
`plot_similarity_gap_bars` compares the MIR values with the true network and shows whether a threshold can separate true edges from non-edges.

## Example

```python
import numpy as np
from complex_network_dynamics import (
    load_network, logistic, df_logistic, simulate_discrete_models, MIR, plot_similarity_gap_bars,
)

A = load_network("16nodes")
traj, _ = simulate_discrete_models(np.random.rand(16), logistic, df_logistic, A, t=10_000, alpha=0.1)

mir, mi = MIR(traj)
band = plot_similarity_gap_bars(mir, A, title="MIR inference, 16 nodes")
```

Pass `nodes=(i, j)` to `MIR` to plot how MI and MIR for one pair change with the grid size.

## API

::: complex_network_dynamics.network_inference.MIR

::: complex_network_dynamics.network_inference.plot_similarity_gap_bars

::: complex_network_dynamics.network_inference.Grid

::: complex_network_dynamics.network_inference.corr_decay

::: complex_network_dynamics.network_inference.compute_single_pair_mi

::: complex_network_dynamics.network_inference.location_jit
