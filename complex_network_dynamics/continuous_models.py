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