# Network Metrics

Distance metrics for networks that may be disconnected.

- `mode="giant"`: diameter and average shortest path length of the largest connected component.
- `mode="all"`: the largest diameter over all components, and the average path length over every reachable pair of nodes.

```python
import networkx as nx
from complex_network_dynamics import load_network, calculate_distance_metrics

G = nx.from_numpy_array(load_network("12nodes"))
diameter, avg_path = calculate_distance_metrics(G)
```

## API

::: complex_network_dynamics.network_metrics.calculate_distance_metrics
