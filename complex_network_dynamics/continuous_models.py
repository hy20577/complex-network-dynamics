import numpy as np
from numba import njit

#### ## # ## # ## # # njit LORENZ  # # # # # # # # # # ## ## ## ## 

# =======================================================================
# Single Lorenz System
# =======================================================================

@njit
def lorenz_njit(t, y, args):
    """Lorenz vector field, compiled with Numba.

    Use with `build_rk4_lyapunov` or `rk4_lyapunov_fast`.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous).
    y : np.ndarray
        State [x, y, z].
    args : tuple
        (sigma, rho, beta).

    Returns
    -------
    np.ndarray
        Flat array [dx/dt, dy/dt, dz/dt] of shape (3,).
    """
    sigma, rho, beta = args
    x, y_coord, z = y
    
    dxdt = sigma * (y_coord - x)
    dydt = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z
    
    # Must return a flat 1D array, no .reshape((1,3))
    return np.array([dxdt, dydt, dzdt], dtype=np.float64)

@njit
def lorenz_jacobian_njit(t, y, args):
    """Jacobian of the Lorenz system, compiled with Numba.

    Parameters
    ----------
    t : float
        Time (unused).
    y : np.ndarray
        State [x, y, z].
    args : tuple
        (sigma, rho, beta).

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3, 3).
    """
    sigma, rho, beta = args
    x, y_coord, z = y
    
    return np.array([
        [-sigma, sigma, 0.0], 
        [rho - z, -1.0, -x], 
        [y_coord, x, -beta]
    ], dtype=np.float64)
    
# =======================================================================
# Coupled Lorenz System & Jacobian
# =======================================================================

@njit
def coupled_lorenz_njit(t, y, args):
    """Network of diffusively coupled Lorenz oscillators, compiled with Numba.

    Nodes are coupled through the x variable:
    dx_i/dt = sigma (y_i - x_i) + K * sum_j A_ij (x_j - x_i).

    Parameters
    ----------
    t : float
        Time (unused).
    y : np.ndarray
        Blocked state vector of length 3N: [x_1..x_N, y_1..y_N, z_1..z_N].
    args : tuple
        (A_mat, K, sigma, rho, beta), where A_mat is the (N, N) adjacency
        matrix and K the coupling strength.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3N.
    """
    A_mat, K, sigma, rho, beta = args
    N = A_mat.shape[0]

    # Unpack blocked components: [x_all, y_all, z_all]
    x = y[0:N]
    y_coord = y[N : 2 * N]
    z = y[2 * N : 3 * N]

    # Diffusive coupling term
    degrees = np.sum(A_mat, axis=1)
    coupling_x = K * (A_mat @ x - degrees * x)

    # Compute node derivatives
    dxdt = sigma * (y_coord - x) + coupling_x
    dydt_nodes = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z

    # Fast Numba array construction (avoids np.concatenate overhead)
    dy = np.empty(3 * N, dtype=np.float64)
    dy[0:N] = dxdt
    dy[N:2*N] = dydt_nodes
    dy[2*N:3*N] = dzdt
    
    return dy
    
@njit
def coupled_lorenz_jacobian_njit(t, y, args):
    """Jacobian of the coupled Lorenz network, compiled with Numba.

    Parameters
    ----------
    t : float
        Time (unused).
    y : np.ndarray
        Blocked state vector of length 3N.
    args : tuple
        (A_mat, K, sigma, rho, beta).

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3N, 3N).
    """
    A_mat, K, sigma, rho, beta = args
    N = A_mat.shape[0]
    
    x = y[0:N]
    y_coord = y[N : 2 * N]
    z = y[2 * N : 3 * N]

    degrees = np.sum(A_mat, axis=1)

    # Allocate full 3N x 3N Jacobian
    J = np.zeros((3 * N, 3 * N), dtype=np.float64)

    # dx/dx block: -sigma * I + coupling
    J[0:N, 0:N] = K * A_mat - np.diag(sigma + K * degrees)

    # dx/dy block: sigma * I
    J[0:N, N : 2 * N] = sigma * np.eye(N)

    # dy/dx block: diag(rho - z)
    J[N : 2 * N, 0:N] = np.diag(rho - z)

    # dy/dy block: -I
    J[N : 2 * N, N : 2 * N] = -np.eye(N)

    # dy/dz block: diag(-x)
    J[N : 2 * N, 2 * N : 3 * N] = np.diag(-x)

    # dz/dx block: diag(y_coord)
    J[2 * N : 3 * N, 0:N] = np.diag(y_coord)

    # dz/dy block: diag(x)
    J[2 * N : 3 * N, N : 2 * N] = np.diag(x)

    # dz/dz block: -beta * I
    J[2 * N : 3 * N, 2 * N : 3 * N] = -beta * np.eye(N)

    return J




#### ## # ## # ## # # LOTKA - VOLTERRA  # # # # # # # # # # ## ## ## ## 

def lotka_volterra(t, y, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    """Lotka-Volterra predator-prey model.

    dx/dt = alpha x - beta x p,   dp/dt = delta x p - gamma p

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        State [x, p]: y[0] is the prey, y[1] the predator population.
    alpha : float, optional
        Prey growth rate. Default is 1.1.
    beta : float, optional
        Predation rate. Default is 0.4.
    delta : float, optional
        Predator growth per prey eaten. Default is 0.1.
    gamma : float, optional
        Predator death rate. Default is 0.4.

    Returns
    -------
    np.ndarray
        [dx/dt, dp/dt].
    """
    x, p = y[0], y[1]
    dxdt = alpha * x - beta * x * p
    dpdt = delta * x * p - gamma * p
    return np.array([dxdt, dpdt])

def lotka_volterra_jacobian(
    t, y, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    """Jacobian matrix of the 2D Lotka-Volterra predator-prey model.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        State [x, p]: y[0] is the prey, y[1] the predator population.
    alpha, beta, delta, gamma : float, optional
        Model parameters, as in `lotka_volterra`.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (2, 2).
    """
    x, p = y[0], y[1]

    df1_dx = alpha - beta * p
    df1_dp = -beta * x
    df2_dx = delta * p
    df2_dp = delta * x - gamma

    return np.array([[df1_dx, df1_dp], [df2_dx, df2_dp]], dtype=float)


#### ## # ## # ## # # LORENZ  # # # # # # # # # # ## ## ## ## 

def lorenz(t, y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Lorenz system.

    dx/dt = sigma (y - x),  dy/dt = x (rho - z) - y,  dz/dt = x y - beta z

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        State [x, y, z].
    sigma : float, optional
        Prandtl number. Default is 10.0.
    rho : float, optional
        Rayleigh number. Default is 28.0.
    beta : float, optional
        Geometric factor. Default is 8/3.

    Returns
    -------
    np.ndarray
        [dx/dt, dy/dt, dz/dt] with shape (1, 3).
    """
    x, y_coord, z = y
    dxdt = sigma * (y_coord - x)
    dydt = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z
    return np.array([dxdt, dydt, dzdt]).reshape((1,3))

def lorenz_jacobian(t, y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Jacobian of the Lorenz system.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        State [x, y, z].
    sigma : float, optional
        Prandtl number. Default is 10.0.
    rho : float, optional
        Rayleigh number. Default is 28.0.
    beta : float, optional
        Geometric factor. Default is 8/3.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3, 3).
    """
    x, y_coord, z = y
    return np.array(
        [[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y_coord, x, -beta]]
    )
    
    
## ### ### ### ### Coupled Versions for Lorenz and Its Jacobian ### ### ###

def coupled_lorenz(t, y, A, K=0.1, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Network of diffusively coupled Lorenz oscillators.

    Nodes are coupled through the x variable:
    dx_i/dt = sigma (y_i - x_i) + K * sum_j A_ij (x_j - x_i).

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3N: [x_1..x_N, y_1..y_N, z_1..z_N].
    A : array_like
        Adjacency matrix of shape (N, N).
    K : float, optional
        Coupling strength. Default is 0.1.
    sigma : float, optional
        Prandtl number. Default is 10.0.
    rho : float, optional
        Rayleigh number. Default is 28.0.
    beta : float, optional
        Geometric factor. Default is 8/3.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3N.

    Raises
    ------
    ValueError
        If the size of `y` is not 3N.
    """
    y_flat = np.asarray(y, dtype=float).ravel()
    A_mat = np.asarray(A, dtype=float)

    N = A_mat.shape[0]
    if y_flat.size != 3 * N:
        raise ValueError(
            f"Dimension mismatch: A is ({N}, {N}) expecting 3*N={3*N} states, but got y of size {y_flat.size}"
        )

    # 1. Unpack blocked components: [x_all, y_all, z_all]
    x = y_flat[0:N]
    y_coord = y_flat[N : 2 * N]
    z = y_flat[2 * N : 3 * N]

    # Diffusive coupling term: K * sum_j A_ij (x_j - x_i)
    degrees = np.asarray(A_mat.sum(axis=1)).ravel()
    coupling_x = K * (A_mat @ x - degrees * x)

    # Compute node derivatives
    dxdt = sigma * (y_coord - x) + coupling_x
    dydt_nodes = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z

    # 2. Return as blocked 1D vector: [dx, dy, dz]
    return np.concatenate([dxdt, dydt_nodes, dzdt])
    
    
def coupled_lorenz_jacobian(
    t, y, A, K=0.1, sigma=10.0, rho=28.0, beta=8.0 / 3.0
):
    """Jacobian of the coupled Lorenz network.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3N: [x_1..x_N, y_1..y_N, z_1..z_N].
    A : array_like
        Adjacency matrix of shape (N, N).
    K : float, optional
        Coupling strength. Default is 0.1.
    sigma : float, optional
        Prandtl number. Default is 10.0.
    rho : float, optional
        Rayleigh number. Default is 28.0.
    beta : float, optional
        Geometric factor. Default is 8/3.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3N, 3N).
    """
    y_flat = np.asarray(y, dtype=float).ravel()
    A_mat = np.asarray(A, dtype=float)

    N = A_mat.shape[0]
    x = y_flat[0:N]
    y_coord = y_flat[N : 2 * N]
    z = y_flat[2 * N : 3 * N]

    degrees = np.asarray(A_mat.sum(axis=1)).ravel()

    # Allocate full 3N x 3N Jacobian
    J = np.zeros((3 * N, 3 * N), dtype=float)

    # Slice indices:
    # x-block: [0 : N], y-block: [N : 2N], z-block: [2N : 3N]

    # dx/dx block: -sigma * I + coupling
    J[0:N, 0:N] = K * A_mat - np.diag(sigma + K * degrees)

    # dx/dy block: sigma * I
    J[0:N, N : 2 * N] = sigma * np.eye(N)

    # dy/dx block: diag(rho - z)
    J[N : 2 * N, 0:N] = np.diag(rho - z)

    # dy/dy block: -I
    J[N : 2 * N, N : 2 * N] = -np.eye(N)

    # dy/dz block: diag(-x)
    J[N : 2 * N, 2 * N : 3 * N] = np.diag(-x)

    # dz/dx block: diag(y_coord)
    J[2 * N : 3 * N, 0:N] = np.diag(y_coord)

    # dz/dy block: diag(x)
    J[2 * N : 3 * N, N : 2 * N] = np.diag(x)

    # dz/dz block: -beta * I
    J[2 * N : 3 * N, 2 * N : 3 * N] = -beta * np.eye(N)

    return J

#### ## # ## # ## # # KURAMOTO  # # # # # # # # # # ## ## ## ##

def kuramoto(t, y, A, w, K=1.0):
    """Kuramoto oscillators on a network.

    dtheta_i/dt = w_i + (K / M) * sum_j A_ij sin(theta_j - theta_i)

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Phases theta_1..theta_M.
    A : array_like
        Adjacency matrix, shape (M, M).
    w : array_like
        Natural frequencies, shape (M,).
    K : float, optional
        Coupling strength. Default is 1.0.

    Returns
    -------
    np.ndarray
        dtheta/dt, shape (M,).
    """
    theta = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    diff = theta[None, :] - theta[:, None]  # theta_j - theta_i
    return np.asarray(w, dtype=float) + (K / len(theta)) * np.sum(A * np.sin(diff), axis=1)


def kuramoto_jacobian(t, y, A, w, K=1.0):
    """Jacobian of `kuramoto`, shape (M, M)."""
    theta = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = len(theta)
    J = (K / M) * A * np.cos(theta[None, :] - theta[:, None])
    np.fill_diagonal(J, 0.0)  # a self-loop adds sin(0), which has zero derivative
    J -= np.diag(J.sum(axis=1))
    return J


def stochastic_kuramoto(theta0, A, w, K=10.0, D=10.0, dt=0.1, n_steps=1000, rng=None):
    """Stochastic Kuramoto oscillators, integrated with Euler-Maruyama.

    dtheta_i = [w_i + (K / M) sum_j A_ij sin(theta_j - theta_i)] dt + D dW_i,

    with independent Wiener processes W_i. This is the model of the thesis,
    integrated with the Euler-Maruyama scheme.

    Parameters
    ----------
    theta0 : array_like
        Initial phases, shape (M,).
    A : array_like
        Adjacency matrix, shape (M, M).
    w : array_like
        Natural frequencies, shape (M,).
    K : float, optional
        Coupling strength. Default is 10, as in the thesis.
    D : float, optional
        Noise strength. Default is 10, as in the thesis.
    dt : float, optional
        Time step. Default is 0.1.
    n_steps : int, optional
        Number of steps. Default is 1000.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    t : np.ndarray
        Times, shape (n_steps + 1,).
    phases : np.ndarray
        Phases (not wrapped), shape (n_steps + 1, M); the first row is theta0.
    """
    rng = np.random.default_rng(rng)
    theta = np.asarray(theta0, dtype=float).copy()
    phases = np.empty((n_steps + 1, len(theta)))
    phases[0] = theta
    noise = D * np.sqrt(dt) * rng.standard_normal((n_steps, len(theta)))
    for k in range(n_steps):
        theta = theta + kuramoto(0.0, theta, A, w, K) * dt + noise[k]
        phases[k + 1] = theta
    return dt * np.arange(n_steps + 1), phases


#### ## # ## # ## # # ROSSLER  # # # # # # # # # # ## ## ## ##

def rossler(t, y, A, K=0.1, w=1.0, a=0.2, f=0.2, c=5.7):
    """Rossler oscillators coupled through x.

    dx_i/dt = -w_i y_i - z_i + K * sum_j A_ij (x_j - x_i)
    dy_i/dt =  w_i x_i + a y_i
    dz_i/dt =  f + z_i (x_i - c)

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [x_1..x_M, y_1..y_M, z_1..z_M].
    A : array_like
        Adjacency matrix, shape (M, M).
    K : float, optional
        Coupling strength. Default is 0.1.
    w : float or array_like, optional
        Frequency of each oscillator. Default is 1.0.
    a, f, c : float, optional
        Rossler parameters. Defaults are 0.2, 0.2 and 5.7.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3M.
    """
    s = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    x, yy, z = s[:M], s[M:2 * M], s[2 * M:]
    w = np.broadcast_to(np.asarray(w, dtype=float), (M,))
    coupling = K * (A @ x - A.sum(axis=1) * x)
    return np.concatenate([-w * yy - z + coupling, w * x + a * yy, f + z * (x - c)])


def rossler_jacobian(t, y, A, K=0.1, w=1.0, a=0.2, f=0.2, c=5.7):
    """Jacobian of `rossler`, shape (3M, 3M)."""
    s = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    x, z = s[:M], s[2 * M:]
    w = np.broadcast_to(np.asarray(w, dtype=float), (M,))
    I = np.eye(M)
    J = np.zeros((3 * M, 3 * M))
    J[:M, :M] = K * A - np.diag(K * A.sum(axis=1))
    J[:M, M:2 * M] = -np.diag(w)
    J[:M, 2 * M:] = -I
    J[M:2 * M, :M] = np.diag(w)
    J[M:2 * M, M:2 * M] = a * I
    J[2 * M:, :M] = np.diag(z)
    J[2 * M:, 2 * M:] = np.diag(x - c)
    return J


#### ## # ## # ## # # HINDMARSH - ROSE  # # # # # # # # # # ## ## ## ##

def hindmarsh_rose(t, y, A, g=0.1):
    """Hindmarsh-Rose neurons coupled through the membrane potential.

    dp_i/dt = q_i - p_i^3 + 3 p_i^2 - n_i + 3.25 - g * sum_l C_il p_l
    dq_i/dt = 1 - 5 p_i^2 - q_i
    dn_i/dt = 0.005 * (4 (p_i + 1.6) - n_i)

    where C = D - A is the graph Laplacian (D the diagonal degree matrix).

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [p_1..p_M, q_1..q_M, n_1..n_M].
    A : array_like
        Adjacency matrix, shape (M, M).
    g : float, optional
        Coupling strength. Default is 0.1.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3M.
    """
    s = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    p, q, n = s[:M], s[M:2 * M], s[2 * M:]
    C = np.diag(A.sum(axis=1)) - A
    return np.concatenate([
        q - p**3 + 3 * p**2 - n + 3.25 - g * (C @ p),
        1 - 5 * p**2 - q,
        0.02 * (p + 1.6) - 0.005 * n,
    ])


def hindmarsh_rose_jacobian(t, y, A, g=0.1):
    """Jacobian of `hindmarsh_rose`, shape (3M, 3M).

    The entry dp/dp is the derivative of the vector field, -3 p^2 + 6 p.
    """
    s = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    p = s[:M]
    C = np.diag(A.sum(axis=1)) - A
    I = np.eye(M)
    J = np.zeros((3 * M, 3 * M))
    J[:M, :M] = np.diag(-3 * p**2 + 6 * p) - g * C
    J[:M, M:2 * M] = I
    J[:M, 2 * M:] = -I
    J[M:2 * M, :M] = np.diag(-10 * p)
    J[M:2 * M, M:2 * M] = -I
    J[2 * M:, :M] = 0.02 * I
    J[2 * M:, 2 * M:] = -0.005 * I
    return J
