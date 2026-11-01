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

## Deciding which links are real: surrogate test

A high MIR alone does not say where to draw the line between links and non-links. `mir_surrogate_test` compares the MIR of each pair with its MIR on \(1/p_{sl}\) [surrogate datasets](surrogates.md):

1. The p-value of a pair is the fraction of surrogates whose MIR is at least the MIR of the data.
2. The Benjamini–Hochberg procedure (`fdr_bh`) controls the false discovery rate, with \(q = p_{sl}/100\) by default, as in the thesis code.
3. If the true network is known, `tpr_fpr` gives the true and false positive rates.

The test can be repeated for growing data lengths (`time_range`) to see how much data the inference needs.

## More tools from the thesis code

| Function | MATLAB name | What it does |
|----------|-------------|--------------|
| `double_normalised_mir` | `double_normalised_MIR` | MIR min-max scaled per grid size, summed over grid sizes and scaled to a maximum of 1; with the true network, the best reconstruction rate over thresholds |
| `lagged_double_mir` | `lagged_double_MIR` | As above, but each pair uses its largest MI over time lags \(-L,\dots,L\), for links with a delay |
| `mutual_information` | `Mutual_info` | Plain MI on a fixed \(N_x \times N_y\) grid, with Pearson's chi-square measure of independence |
| `best_case_tpr_fpr` | `best_case_TPR_FPR` | ROC sweep over all thresholds of a similarity matrix, and the threshold closest to perfect inference |

```python
from complex_network_dynamics import double_normalised_mir, best_case_tpr_fpr, load_network

A = load_network("16nodes")
normalised, max_rate = double_normalised_mir(traj, true_network=A)
roc, best, inferred = best_case_tpr_fpr(normalised, A)
best   # TPR, FPR, distance to (0, 1), threshold
```

!!! note "Matching the thesis code"
    These functions reproduce the MATLAB code exactly, including its details:
    `lagged_double_mir` uses the slightly different grid-size rule written inside
    `lagged_double_MIR.m`, and `mutual_information` drops points at each node's
    maximum and returns MATLAB's `gammainc(v/2, chi2/2)` as `P_null`. The standard
    chi-square p-value would be `scipy.stats.chi2.sf(chi2, v)`.
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

::: complex_network_dynamics.network_inference.mir_surrogate_test

::: complex_network_dynamics.network_inference.fdr_bh

::: complex_network_dynamics.network_inference.tpr_fpr

::: complex_network_dynamics.network_inference.double_normalised_mir

::: complex_network_dynamics.network_inference.lagged_double_mir

::: complex_network_dynamics.network_inference.mutual_information

::: complex_network_dynamics.network_inference.best_case_tpr_fpr

::: complex_network_dynamics.network_inference.plot_similarity_gap_bars

::: complex_network_dynamics.network_inference.Grid

::: complex_network_dynamics.network_inference.corr_decay

::: complex_network_dynamics.network_inference.compute_single_pair_mi

::: complex_network_dynamics.network_inference.location_jit
