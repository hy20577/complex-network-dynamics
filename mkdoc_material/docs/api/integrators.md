# Integrators

Integrate a system and compute its **Lyapunov spectrum** at the same time.

Alongside the state \(y\), each integrator evolves an orthonormal tangent frame \(Q\) with the variational equation
\(\dot Q = J(t, y)\,Q\). Every few steps \(Q\) is re-orthonormalized by a QR decomposition \(Q = Q' R\), and the
Lyapunov exponents are the time-averaged logarithms of the diagonal of \(R\):

\[
\lambda_i = \lim_{T\to\infty} \frac{1}{T} \sum_k \ln |R^{(k)}_{ii}|
\]

## Which integrator?

| Function | Method | Speed | Needs |
|----------|--------|-------|-------|
| `euler_with_lyapunov` | Forward Euler, fixed step | fast, low accuracy | NumPy model |
| `rk4_with_lyapunov` | RK4, fixed step | good default | NumPy model |
| `solve_ivp_with_lyapunov` | SciPy adaptive (`RK45`, `DOP853`, …) | slower, most accurate | NumPy model |
| `rk4_lyapunov_fast` | RK4, Numba-compiled | very fast | Numba (`*_njit`) model |
| `build_rk4_lyapunov` | RK4, compiled per system, with transient | fastest | Numba (`*_njit`) model |

## The `lyap_mode` option

The NumPy integrators accept `lyap_mode`:

| `lyap_mode` | Returns |
|-------------|---------|
| `"none"` | `t_vals, traj` |
| `"spectrum"` (default) | `t_vals, traj, spectrum` |
| `"history"` | `t_vals, traj, lyap_t, lyap_history` |

Use `"history"` to check that the exponents have converged.

## Examples

```python
import numpy as np
from complex_network_dynamics import lorenz, lorenz_jacobian, rk4_with_lyapunov

t, traj, spectrum = rk4_with_lyapunov(
    lorenz, y0=np.ones(3), t0=0.0, h=0.01, n_steps=50_000, df=lorenz_jacobian,
)
spectrum   # approximately [0.9, 0.0, -14.6]
```

Compiled version:

```python
import numpy as np
from complex_network_dynamics import build_rk4_lyapunov, lorenz_njit, lorenz_jacobian_njit

solver = build_rk4_lyapunov(lorenz_njit, lorenz_jacobian_njit)
t, traj, spectrum, lyap_t, lyap_hist = solver(
    np.ones(3), 0.0, 0.01, 50_000, 10, (10.0, 28.0, 8.0 / 3.0)
)
```

## API

::: complex_network_dynamics.integrators.euler_with_lyapunov

::: complex_network_dynamics.integrators.rk4_with_lyapunov

::: complex_network_dynamics.integrators.solve_ivp_with_lyapunov

::: complex_network_dynamics.integrators.rk4_lyapunov_fast

::: complex_network_dynamics.integrators.build_rk4_lyapunov

::: complex_network_dynamics.integrators._apply_qr_update
