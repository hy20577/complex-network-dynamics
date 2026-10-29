# Discrete Models

One-dimensional chaotic maps and networks of coupled maps.

## Local maps

**Logistic map**

\[
f(x) = r\,x\,(1 - x), \qquad f'(x) = r\,(1 - 2x)
\]

**Circle map**

\[
f(x) = x + r - \frac{K}{2\pi}\sin(2\pi x) \pmod 1, \qquad f'(x) = 1 - K\cos(2\pi x)
\]

## Coupled map network

`simulate_discrete_models` couples \(M\) maps on a network with adjacency matrix \(A\).
With \(k_i\) the degree of node \(i\) and coupling strength \(\alpha\):

\[
x_i(n+1) = (1-\alpha)\, f\big(x_i(n)\big) + \frac{\alpha}{k_i}\sum_{j} A_{ij}\, f\big(x_j(n)\big)
\]

Isolated nodes keep their own dynamics. The full Lyapunov spectrum comes from the Jacobian
\(J = \big[(1-\alpha) I + \alpha W\big]\,\mathrm{diag}\big(f'(x)\big)\) with QR re-orthonormalization at every step.

```python
import numpy as np
from complex_network_dynamics import load_network, logistic, df_logistic, simulate_discrete_models

A = load_network("16nodes")
x0 = np.random.rand(A.shape[0])

traj, lyap = simulate_discrete_models(x0, logistic, df_logistic, A, t=10_000, alpha=0.1, r=4.0)
traj.shape   # (10000, 16)
lyap[:3]     # largest Lyapunov exponents
```

Extra keyword arguments (here `r=4.0`) go to both `f` and `df`.

## API

::: complex_network_dynamics.discrete_models.logistic

::: complex_network_dynamics.discrete_models.df_logistic

::: complex_network_dynamics.discrete_models.circle

::: complex_network_dynamics.discrete_models.df_circle

::: complex_network_dynamics.discrete_models.simulate_discrete_models
