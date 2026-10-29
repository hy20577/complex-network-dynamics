# Datasets

Example networks ship with the package in the `complex_network_dynamics/Data` folder. Each file is a tab-separated adjacency matrix.

| Name      | Nodes | Type                     |
|-----------|-------|--------------------------|
| `12nodes` | 12    | Undirected, unweighted   |
| `16nodes` | 16    | Undirected, unweighted   |

```python
from complex_network_dynamics import load_network

A = load_network("16nodes")      # or "16nodes.txt"
A.shape                          # (16, 16)
```

!!! tip "Adding your own network"
    Put a tab-separated `.txt` file in `complex_network_dynamics/Data/` and load it by its file name.
    Everything in `Data/` is bundled when the package is built.

## API

::: complex_network_dynamics.datasets.load_network
