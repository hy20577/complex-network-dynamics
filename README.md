# complex-network-dynamics

Simulation and stability analysis of nonlinear dynamical systems and complex networks.

The package provides:

- **Discrete-time models**: logistic and circle maps, plus coupled-map networks with the full Lyapunov spectrum.
- **Continuous-time models**: Lorenz, coupled Lorenz networks and Lotka–Volterra, with their Jacobians (plain NumPy and Numba-compiled versions).
- **Integrators with Lyapunov exponents**: Euler, RK4, SciPy `solve_ivp`, and a fast Numba RK4.
- **Network inference**: Mutual Information Rate (MIR) to reconstruct a network from node time series, plus a plot to judge how well the inferred edges separate from non-edges.
- **Network metrics**: diameter and average shortest path length, for connected and disconnected graphs.
- **Built-in datasets**: example adjacency matrices in the `Data` folder.

## Installation

Requires Python 3.10–3.13 (the upper bound keeps Numba compatible).

From the project root:

```bash
pip install -e .
```

Dependencies (installed automatically): `numpy`, `scipy`, `matplotlib`, `numba`, `networkx`.

## Project layout

```
complex_network_dynamics/
├── pyproject.toml
├── README.md
├── complex_network_dynamics/
│   ├── __init__.py
│   ├── datasets.py            # load_network()
│   ├── discrete_models.py     # logistic, circle maps, coupled-map simulation
│   ├── continuous_models.py   # Lorenz, coupled Lorenz, Lotka–Volterra (+ Jacobians)
│   ├── integrators.py         # Euler / RK4 / solve_ivp with Lyapunov spectrum
│   ├── network_inference.py   # MIR, mutual information, inference plots
│   ├── network_metrics.py     # diameter and average path length
│   └── Data/
│       ├── 12nodes.txt
│       └── 16nodes.txt
└── mkdoc_material/            # MkDocs documentation site
```

## Datasets

The `Data` folder holds tab-separated adjacency matrices that ship with the package:

| Name      | Nodes | Description                          |
|-----------|-------|--------------------------------------|
| `12nodes` | 12    | Undirected, unweighted example network |
| `16nodes` | 16    | Undirected, unweighted example network |

Load one by name (the `.txt` extension is optional):

```python
from complex_network_dynamics import load_network

A = load_network("16nodes")   # numpy array, shape (16, 16)
```

To add your own network, put a tab-separated `.txt` file in `complex_network_dynamics/Data/` and load it by its file name. Files in `Data/` are bundled into the package when it is built (see `[tool.setuptools.package-data]` in `pyproject.toml`).

## Quick start

### Coupled logistic maps on a network

```python
import numpy as np
from complex_network_dynamics import load_network, logistic, df_logistic, simulate_discrete_models

A = load_network("16nodes")
x0 = np.random.rand(A.shape[0])

traj, lyap = simulate_discrete_models(x0, logistic, df_logistic, A, t=10_000, alpha=0.1, r=4.0)
print(traj.shape)   # (10000, 16)
print(lyap)         # Lyapunov exponents, largest first
```

### Lorenz system with Lyapunov spectrum

```python
import numpy as np
from complex_network_dynamics import lorenz, lorenz_jacobian, rk4_with_lyapunov

t_vals, traj, spectrum = rk4_with_lyapunov(
    lorenz, y0=np.array([1.0, 1.0, 1.0]), t0=0.0, h=0.01, n_steps=50_000,
    df=lorenz_jacobian, lyap_mode="spectrum",
)
```

`lyap_mode` can be `"none"` (returns `t_vals, traj`), `"spectrum"` (returns `t_vals, traj, spectrum`) or `"history"` (returns `t_vals, traj, lyap_t, lyap_history`). Model parameters (e.g. `sigma`, `rho`, `beta`) are passed as keyword arguments.

### Inferring a network with MIR

```python
from complex_network_dynamics import MIR, plot_similarity_gap_bars

# traj: (time steps, nodes) array, e.g. from simulate_discrete_models
mir_matrix, mi_matrix = MIR(traj)   # MIR and plain mutual information, both (nodes, nodes)
plot_similarity_gap_bars(mir_matrix, A, title="MIR inference, 16 nodes")
```

### Network distance metrics

```python
import networkx as nx
from complex_network_dynamics import load_network, calculate_distance_metrics

G = nx.from_numpy_array(load_network("12nodes"))
diameter, avg_path = calculate_distance_metrics(G, mode="giant")
```

## Documentation

API documentation is built with MkDocs Material and mkdocstrings from `mkdoc_material/`. Each module has its own page, and the function reference is generated from the docstrings.

```bash
cd mkdoc_material && ./venv/bin/python -m mkdocs serve
```

The `venv` folder is not kept in Git. To create it on a new machine:

```bash
cd mkdoc_material && python3 -m venv venv && ./venv/bin/python -m pip install -r requirements.txt
```
