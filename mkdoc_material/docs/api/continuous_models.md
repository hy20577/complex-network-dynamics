# Continuous Models

Vector fields and Jacobians for continuous-time systems. Each model comes in two flavours:

| Flavour | Signature | Use with |
|---------|-----------|----------|
| NumPy   | `f(t, y, **params)` | `euler_with_lyapunov`, `rk4_with_lyapunov`, `solve_ivp_with_lyapunov` |
| Numba (`*_njit`) | `f(t, y, args)` with `args` a tuple | `build_rk4_lyapunov`, `rk4_lyapunov_fast` |

## Lorenz system

\[
\dot x = \sigma (y - x), \qquad \dot y = x(\rho - z) - y, \qquad \dot z = x y - \beta z
\]

Defaults \(\sigma = 10\), \(\rho = 28\), \(\beta = 8/3\) give the chaotic Lorenz attractor.

## Coupled Lorenz network

\(N\) Lorenz oscillators are coupled diffusively through \(x\), with adjacency matrix \(A\) and strength \(K\):

\[
\dot x_i = \sigma (y_i - x_i) + K \sum_j A_{ij}\,(x_j - x_i)
\]

The state is a blocked vector of length \(3N\): \([x_1,\dots,x_N,\; y_1,\dots,y_N,\; z_1,\dots,z_N]\).

```python
import numpy as np
from complex_network_dynamics import load_network, coupled_lorenz, coupled_lorenz_jacobian, rk4_with_lyapunov

A = load_network("12nodes")
y0 = np.random.rand(3 * A.shape[0])

t, traj, spectrum = rk4_with_lyapunov(
    coupled_lorenz, y0, t0=0.0, h=0.01, n_steps=20_000,
    df=coupled_lorenz_jacobian, A=A, K=0.5,
)
```

## Lotka–Volterra

\[
\dot x = \alpha x - \beta x p, \qquad \dot p = \delta x p - \gamma p
\]

with \(x\) the prey and \(p\) the predator population.

## API

### NumPy versions

::: complex_network_dynamics.continuous_models.lorenz

::: complex_network_dynamics.continuous_models.lorenz_jacobian

::: complex_network_dynamics.continuous_models.coupled_lorenz

::: complex_network_dynamics.continuous_models.coupled_lorenz_jacobian

::: complex_network_dynamics.continuous_models.lotka_volterra

::: complex_network_dynamics.continuous_models.lotka_volterra_jacobian

### Numba versions

::: complex_network_dynamics.continuous_models.lorenz_njit

::: complex_network_dynamics.continuous_models.lorenz_jacobian_njit

::: complex_network_dynamics.continuous_models.coupled_lorenz_njit

::: complex_network_dynamics.continuous_models.coupled_lorenz_jacobian_njit
