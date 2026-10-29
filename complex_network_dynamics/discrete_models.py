## Functions

import math

import numpy as np

def df_logistic(x, r=4.0):
    """Derivative of the logistic map: f'(x) = r * (1 - 2x)

    Parameters
    ----------
    x : float or np.ndarray
        State(s) at which to evaluate the derivative.
    r : float, optional
        Growth parameter. Default is 4.0.

    Returns
    -------
    float or np.ndarray
        f'(x), with the same shape as `x`.
    """
    return r * (1.0 - 2.0 * x)

def df_circle(x, r=0.35, K=6.9115):
    """Derivative of the circle map: f'(x) = 1 - K * cos(2 * pi * x)

    Parameters
    ----------
    x : float or np.ndarray
        State(s) at which to evaluate the derivative.
    r : float, optional
        Rotation parameter (unused by the derivative, kept so `f` and `df`
        accept the same keyword arguments). Default is 0.35.
    K : float, optional
        Nonlinearity strength. Default is 6.9115.

    Returns
    -------
    float or np.ndarray
        f'(x), with the same shape as `x`.
    """
    return 1.0 - K * np.cos(2.0 * np.pi * x)
    
def logistic(x0, r=4):
    """Logistic map: f(x) = r * x * (1 - x)

    Parameters
    ----------
    x0 : float or array_like
        Current state(s), usually in [0, 1].
    r : float, optional
        Growth parameter. r = 4 gives fully developed chaos. Default is 4.

    Returns
    -------
    np.ndarray
        Next state(s), with the same shape as `x0`.
    """
    x0 = np.asarray(x0)
    y = r*x0*(1-x0)
    return y
    
def circle(x0, r=0.35, K=6.9115):
    """Circle map: f(x) = x + r - (K / 2pi) * sin(2 * pi * x)  (mod 1)

    Parameters
    ----------
    x0 : float or array_like
        Current state(s) in [0, 1).
    r : float, optional
        Rotation parameter. Default is 0.35.
    K : float, optional
        Nonlinearity strength. Default is 6.9115.

    Returns
    -------
    np.ndarray
        Next state(s) in [0, 1), with the same shape as `x0`.
    """
    x0 = np.asarray(x0)
    y= x0+r-(K/(2*math.pi))*np.sin(2*math.pi*x0)
    y = y%1
    return y

def simulate_discrete_models(x0, f, df, A, t, alpha=0.1, n_transient=1000, **params):
    """
    Simulates coupled map trajectories and calculates the full Lyapunov spectrum.

    Parameters
    ----------
    x0          : 1D array of initial node states (shape: (M,))
    f           : Local map function f(x, **params)
    df          : Derivative of local map df(x, **params)
    A           : Adjacency matrix (shape: (M, M))
    t           : Total recording steps (after transients)
    alpha       : Coupling strength in [0, 1]
    n_transient : Warm-up iterations to let trajectory reach the attractor
    **params    : Extra parameters passed to f and df

    Returns
    -------
    traj        : 2D array (t, M) of node states
    lyap_exp    : 1D array (M,) of Lyapunov exponents sorted in descending order
    """
    x = np.asarray(x0, dtype=float).copy()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]

    # Precompute row-normalized transition matrix W
    degree = np.sum(A, axis=1)
    degree_safe = np.where(degree == 0, 1.0, degree)
    W = A / degree_safe[:, np.newaxis]

    # Preserve state for isolated nodes
    isolated_mask = (degree == 0)
    if np.any(isolated_mask):
        W[isolated_mask, :] = 0.0
        W[isolated_mask, isolated_mask] = 1.0

    # Effective linear coupling operator: M_net = (1 - alpha) * I + alpha * W
    M_net = (1.0 - alpha) * np.eye(M) + alpha * W

    # 1. Transient burn-in (lets state settle onto the attractor)
    for _ in range(n_transient):
        f_x = f(x, **params)
        x = M_net @ f_x

    # 2. Main loop: recording trajectories and tracking perturbations
    traj = np.zeros((t, M), dtype=float)
    traj[0] = x

    # Initialize orthonormal tangent frame Q and running sum of log expansion rates
    Q = np.eye(M, dtype=float)
    lyap_sum = np.zeros(M, dtype=float)

    for step in range(1, t):
        # Local state and local derivatives
        f_x = f(x, **params)
        f_prime = df(x, **params)  # shape: (M,)

        # Next state
        x = M_net @ f_x
        traj[step] = x

        # Jacobian J = M_net @ diag(f')
        # Efficient column-scaling: M_net * f_prime avoids an explicit diag() allocation
        J = M_net * f_prime

        # Propagate tangent vectors: Z = J @ Q
        Z = J @ Q

        # Re-orthonormalize via QR decomposition
        Q, R = np.linalg.qr(Z)

        # Accumulate log expansion factors (diagonal elements of upper-triangular R)
        # abs avoids negative values from possible reflection components
        lyap_sum += np.log(np.abs(np.diag(R)))

    # Compute average rates of divergence
    lyap_exp = lyap_sum / (t - 1)
    lyap_exp = np.sort(lyap_exp)[::-1]  # Sort descending (lambda_1 >= lambda_2 >= ...)

    return traj, lyap_exp