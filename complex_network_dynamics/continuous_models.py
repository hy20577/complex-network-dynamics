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

    See Also
    --------
    lorenz : The same system in plain Python, with keyword parameters.
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

    See Also
    --------
    lorenz_jacobian : The same Jacobian in plain Python, with keyword parameters.
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

        dx_i/dt = sigma (y_i - x_i) + K sum_j A_ij (x_j - x_i)
        dy_i/dt = x_i (rho - z_i) - y_i
        dz_i/dt = x_i y_i - beta z_i

    Parameters
    ----------
    t : float
        Time (unused).
    y : np.ndarray
        Blocked state vector of length 3N: [x_1..x_N, y_1..y_N, z_1..z_N].
    args : tuple
        (A_mat, K, sigma, rho, beta), where A_mat is the (N, N) adjacency
        matrix, with the convention of `coupled_lorenz`, and K the coupling
        strength. A_mat must be a float array: an integer matrix fails to
        compile, so pass `A.astype(float)`.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3N.

    See Also
    --------
    coupled_lorenz : The same system in plain Python, with keyword parameters.
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
        (A_mat, K, sigma, rho, beta), as in `coupled_lorenz_njit`; A_mat
        must be a float array.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3N, 3N).

    See Also
    --------
    coupled_lorenz_jacobian : The same Jacobian in plain Python, with keyword parameters.
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

        dx/dt = alpha x - beta x p
        dp/dt = delta x p - gamma p

    The orbits are closed curves around the equilibrium (gamma/delta, alpha/beta),
    on which V = delta x - gamma ln x + beta p - alpha ln p stays constant, so
    the system is not chaotic and both Lyapunov exponents are zero.

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

        dx/dt = sigma (y - x)
        dy/dt = x (rho - z) - y
        dz/dt = x y - beta z

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
        [dx/dt, dy/dt, dz/dt] with shape (3,).

    See Also
    --------
    lorenz_njit : Numba-compiled version for `build_rk4_lyapunov`, about 10 times faster.
    """
    x, y_coord, z = y
    dxdt = sigma * (y_coord - x)
    dydt = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z
    return np.array([dxdt, dydt, dzdt])

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

    See Also
    --------
    lorenz_jacobian_njit : Numba-compiled version for `build_rk4_lyapunov`.
    """
    x, y_coord, z = y
    return np.array(
        [[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y_coord, x, -beta]]
    )
    
    
## ### ### ### ### Coupled Versions for Lorenz and Its Jacobian ### ### ###

def coupled_lorenz(t, y, A, K=0.1, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Network of diffusively coupled Lorenz oscillators.

    Nodes are coupled through the x variable:

        dx_i/dt = sigma (y_i - x_i) + K sum_j A_ij (x_j - x_i)
        dy_i/dt = x_i (rho - z_i) - y_i
        dz_i/dt = x_i y_i - beta z_i

    This is Eq. (20) of Yildirim & Antonopoulos (2024), which uses K = 0.2.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3N: [x_1..x_N, y_1..y_N, z_1..z_N].
    A : array_like
        Adjacency matrix of shape (N, N). A[i, j] is the weight of the link
        from node j to node i: row i lists the nodes that drive node i.
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

    See Also
    --------
    coupled_lorenz_njit : Numba-compiled version for `build_rk4_lyapunov`.
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
        Adjacency matrix of shape (N, N), with the convention of
        `coupled_lorenz`.
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

    Raises
    ------
    ValueError
        If the size of `y` is not 3N.

    See Also
    --------
    coupled_lorenz_jacobian_njit : Numba-compiled version for `build_rk4_lyapunov`.
    """
    y_flat = np.asarray(y, dtype=float).ravel()
    A_mat = np.asarray(A, dtype=float)

    N = A_mat.shape[0]
    if y_flat.size != 3 * N:
        raise ValueError(
            f"Dimension mismatch: A is ({N}, {N}) expecting 3*N={3*N} states, but got y of size {y_flat.size}"
        )
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

def _kuramoto_inputs(y, A, w):
    """Phases, adjacency matrix and frequencies as float arrays of matching size."""
    theta = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    w = np.asarray(w, dtype=float)
    M = theta.size
    if A.shape != (M, M) or w.size != M:
        raise ValueError(
            f"Dimension mismatch: got {M} phases, A of shape {A.shape} and w of size {w.size}"
        )
    return theta, A, w


def kuramoto(t, y, A, w, K=1.0):
    """Kuramoto oscillators on a network.

        dtheta_i/dt = w_i + (K / M) sum_j A_ij sin(theta_j - theta_i)

    This is Eq. (22) of Yildirim & Antonopoulos (2024).

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Phases theta_1..theta_M.
    A : array_like
        Adjacency matrix, shape (M, M). A[i, j] is the weight of the link
        from node j to node i: row i lists the nodes that drive node i.
    w : array_like
        Natural frequencies, shape (M,).
    K : float, optional
        Coupling strength. Default is 1.0.

    Returns
    -------
    np.ndarray
        dtheta/dt, shape (M,).

    Raises
    ------
    ValueError
        If `A` is not (M, M) or `w` does not have M entries.

    See Also
    --------
    stochastic_kuramoto : The same model with noise, Eq. (23) of the paper.
    """
    theta, A, w = _kuramoto_inputs(y, A, w)
    diff = theta[None, :] - theta[:, None]  # theta_j - theta_i
    return w + (K / len(theta)) * np.sum(A * np.sin(diff), axis=1)


def kuramoto_jacobian(t, y, A, w, K=1.0):
    """Jacobian of `kuramoto`.

    J_ij = (K / M) A_ij cos(theta_j - theta_i) for i != j, and each diagonal
    entry is minus the sum of the other entries in its row.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Phases theta_1..theta_M.
    A : array_like
        Adjacency matrix, shape (M, M), with the convention of `kuramoto`.
    w : array_like
        Natural frequencies, shape (M,). They do not enter the Jacobian.
    K : float, optional
        Coupling strength. Default is 1.0.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (M, M).

    Raises
    ------
    ValueError
        If `A` is not (M, M) or `w` does not have M entries.
    """
    theta, A, w = _kuramoto_inputs(y, A, w)
    M = len(theta)
    J = (K / M) * A * np.cos(theta[None, :] - theta[:, None])
    np.fill_diagonal(J, 0.0)  # a self-loop adds sin(0), which has zero derivative
    J -= np.diag(J.sum(axis=1))
    return J


def stochastic_kuramoto(theta0, A, w, K=10.0, D=10.0, dt=0.1, n_steps=1000, rng=None):
    """Stochastic Kuramoto oscillators, integrated with Euler-Maruyama.

        dtheta_i = [w_i + (K / M) sum_j A_ij sin(theta_j - theta_i)] dt + D dW_i

    with independent Wiener processes W_i. This is Eq. (3.18) of the thesis
    and Eq. (23) of Yildirim & Antonopoulos (2024), integrated with the
    Euler-Maruyama scheme. The defaults K = D = 10 are those of the thesis,
    which draws w_i uniformly in [-pi, pi]; the paper uses K = 4, D = 0.05
    and states w_i in [-2 pi, 2 pi].

    Parameters
    ----------
    theta0 : array_like
        Initial phases, shape (M,).
    A : array_like
        Adjacency matrix, shape (M, M), with the convention of `kuramoto`.
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

    Raises
    ------
    ValueError
        If `A` is not (M, M) or `w` does not have M entries.

    See Also
    --------
    kuramoto : The deterministic model, Eq. (22) of the paper.
    """
    rng = np.random.default_rng(rng)
    theta, A, w = _kuramoto_inputs(theta0, A, w)
    theta = theta.copy()
    phases = np.empty((n_steps + 1, len(theta)))
    phases[0] = theta
    noise = D * np.sqrt(dt) * rng.standard_normal((n_steps, len(theta)))
    for k in range(n_steps):
        theta = theta + kuramoto(0.0, theta, A, w, K) * dt + noise[k]
        phases[k + 1] = theta
    return dt * np.arange(n_steps + 1), phases


#### ## # ## # ## # # ROSSLER  # # # # # # # # # # ## ## ## ##

def _rossler_inputs(y, A, w):
    """State, adjacency matrix and frequencies as float arrays of matching size."""
    s = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    w = np.asarray(w, dtype=float)
    if A.shape != (M, M) or s.size != 3 * M or w.size not in (1, M):
        raise ValueError(
            f"Dimension mismatch: A of shape {A.shape} expects 3*M={3 * M} states and 1 or {M} "
            f"frequencies, but got y of size {s.size} and w of size {w.size}"
        )
    return s, A, np.broadcast_to(w.ravel(), (M,)), M


def rossler(t, y, A, K=0.1, w=1.0, a=0.2, f=0.2, c=5.7):
    """Rossler oscillators coupled through x.

        dx_i/dt = -w_i y_i - z_i + K sum_j A_ij (x_j - x_i)
        dy_i/dt =  w_i x_i + a y_i
        dz_i/dt =  f + z_i (x_i - c)

    This is `rossler_ext` of the thesis MATLAB code.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [x_1..x_M, y_1..y_M, z_1..z_M].
    A : array_like
        Adjacency matrix, shape (M, M). A[i, j] is the weight of the link
        from node j to node i: row i lists the nodes that drive node i.
    K : float, optional
        Coupling strength. Default is 0.1.
    w : float or array_like, optional
        Frequency of each oscillator: one value for all, or M values.
        Default is 1.0.
    a, f, c : float, optional
        Rossler parameters. Defaults are 0.2, 0.2 and 5.7.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3M.

    Raises
    ------
    ValueError
        If `y` does not have 3M entries or `w` has neither 1 nor M entries.
    """
    s, A, w, M = _rossler_inputs(y, A, w)
    x, yy, z = s[:M], s[M:2 * M], s[2 * M:]
    coupling = K * (A @ x - A.sum(axis=1) * x)
    return np.concatenate([-w * yy - z + coupling, w * x + a * yy, f + z * (x - c)])


def rossler_jacobian(t, y, A, K=0.1, w=1.0, a=0.2, f=0.2, c=5.7):
    """Jacobian of `rossler`.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [x_1..x_M, y_1..y_M, z_1..z_M].
    A : array_like
        Adjacency matrix, shape (M, M), with the convention of `rossler`.
    K : float, optional
        Coupling strength. Default is 0.1.
    w : float or array_like, optional
        Frequency of each oscillator: one value for all, or M values.
        Default is 1.0.
    a, f, c : float, optional
        Rossler parameters. Defaults are 0.2, 0.2 and 5.7.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3M, 3M).

    Raises
    ------
    ValueError
        If `y` does not have 3M entries or `w` has neither 1 nor M entries.
    """
    s, A, w, M = _rossler_inputs(y, A, w)
    x, z = s[:M], s[2 * M:]
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

def _hindmarsh_rose_inputs(y, A):
    """State and adjacency matrix as float arrays of matching size, and the Laplacian."""
    state = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    M = A.shape[0]
    if A.shape != (M, M) or state.size != 3 * M:
        raise ValueError(
            f"Dimension mismatch: A of shape {A.shape} expects 3*M={3 * M} states, but got y of size {state.size}"
        )
    C = np.diag(A.sum(axis=1)) - A
    return state, C, M


def hindmarsh_rose(t, y, A, g=0.1, a=1.0, b=3.0, c=1.0, d=5.0, r=0.005, s=4.0, p0=-1.6, I_ext=3.25):
    """Hindmarsh-Rose neurons coupled through the membrane potential.

        dp_i/dt = q_i - a p_i^3 + b p_i^2 - n_i + I_ext - g sum_l C_il p_l
        dq_i/dt = c - d p_i^2 - q_i
        dn_i/dt = r (s (p_i - p0) - n_i)

    where C = D - A is the graph Laplacian and D the diagonal matrix of the
    row sums of A, so the coupling is g * sum_l A_il (p_l - p_i). This is
    Eq. (21) of Yildirim & Antonopoulos (2024) and Eq. (3.14) of the thesis,
    as computed by the thesis MATLAB code (`HR_ext`). The paper and thesis
    print C = A - D and r = 0.05 (called h there); the code uses C = D - A
    and r = 0.005, the values used here.

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [p_1..p_M, q_1..q_M, n_1..n_M].
    A : array_like
        Adjacency matrix, shape (M, M). A[i, j] is the weight of the link
        from node j to node i: row i lists the nodes that drive node i.
    g : float, optional
        Coupling strength. Default is 0.1.
    a, b, c, d : float, optional
        Parameters of the fast subsystem. Defaults are 1, 3, 1 and 5.
    r : float, optional
        Time scale of the slow current n (h in the paper). Default is 0.005.
    s, p0 : float, optional
        Adaptation strength and resting potential. Defaults are 4 and -1.6.
    I_ext : float, optional
        External current. Default is 3.25, which gives chaotic bursting.

    Returns
    -------
    np.ndarray
        Blocked derivative vector of length 3M.

    Raises
    ------
    ValueError
        If `A` is not square or `y` does not have 3M entries.
    """
    state, C, M = _hindmarsh_rose_inputs(y, A)
    p, q, n = state[:M], state[M:2 * M], state[2 * M:]
    return np.concatenate([
        q - a * p**3 + b * p**2 - n + I_ext - g * (C @ p),
        c - d * p**2 - q,
        r * (s * (p - p0) - n),
    ])


def hindmarsh_rose_jacobian(t, y, A, g=0.1, a=1.0, b=3.0, c=1.0, d=5.0, r=0.005, s=4.0, p0=-1.6, I_ext=3.25):
    """Jacobian of `hindmarsh_rose`.

    The entry dp/dp is -3 a p^2 + 2 b p - g C, the derivative of the vector
    field (with the defaults, -3 p^2 + 6 p - g C).

    Parameters
    ----------
    t : float
        Time (unused).
    y : array_like
        Blocked state vector of length 3M: [p_1..p_M, q_1..q_M, n_1..n_M].
    A : array_like
        Adjacency matrix, shape (M, M), with the convention of `hindmarsh_rose`.
    g, a, b, c, d, r, s, p0, I_ext : float, optional
        Model parameters, as in `hindmarsh_rose`. c, p0 and I_ext do not
        enter the Jacobian.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (3M, 3M).

    Raises
    ------
    ValueError
        If `A` is not square or `y` does not have 3M entries.
    """
    state, C, M = _hindmarsh_rose_inputs(y, A)
    p = state[:M]
    I = np.eye(M)
    J = np.zeros((3 * M, 3 * M))
    J[:M, :M] = np.diag(-3 * a * p**2 + 2 * b * p) - g * C
    J[:M, M:2 * M] = I
    J[:M, 2 * M:] = -I
    J[M:2 * M, :M] = np.diag(-2 * d * p)
    J[M:2 * M, M:2 * M] = -I
    J[2 * M:, :M] = r * s * I
    J[2 * M:, 2 * M:] = -r * I
    return J
