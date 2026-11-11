import numpy as np
from scipy.signal import hilbert

# Phases and time series are (n_points, n_nodes) arrays, one column per node.


def hilbert_phases(data, demean=True):
    """Instantaneous phases from the Hilbert transform.

    The phase is the angle of the analytic signal x(t) + i H[x](t), measured
    from the origin. The Hilbert transform of a constant is zero, so a signal
    with a nonzero mean gives an orbit shifted away from the origin, and for a
    large offset (e.g. maps on [0, 1]) the angle hardly moves: uncoupled nodes
    then look phase-synchronised. Subtracting each node's mean first removes
    this artefact and changes nothing for signals that oscillate around zero.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes).
    demean : bool, optional
        Subtract each node's mean before the transform. Default is True.
        Use False to reproduce the thesis and Yildirim & Antonopoulos (2024),
        e.g. rho = 0.93 for the 4 logistic maps in Table 1.

    Returns
    -------
    np.ndarray
        Unwrapped phase of each node's analytic signal, same shape as `data`.
    """
    data = np.asarray(data, dtype=float)
    if demean:
        data = data - data.mean(axis=0)
    return np.unwrap(np.angle(hilbert(data, axis=0)), axis=0)


def global_sync(phases):
    """Running average of the Kuramoto order parameter.

    The order parameter at time t is

        rho(t) = |sum_j exp(i theta_j(t))| / M,

    which is 1 when all phases are equal and near 0 when they are spread out.

    Parameters
    ----------
    phases : array_like
        Phases, shape (n_points, M).

    Returns
    -------
    np.ndarray
        Shape (n_points,): the mean of rho over the first 1, 2, ..., n_points
        time points.
    """
    phases = np.asarray(phases, dtype=float)
    rho = np.abs(np.exp(1j * phases).sum(axis=1)) / phases.shape[1]
    return np.cumsum(rho) / np.arange(1, len(rho) + 1)


def pairwise_sync(phases):
    """Phase synchronisation between every pair of nodes.

    r_ij = |mean_t exp(i (theta_i(t) - theta_j(t)))| (Gomez-Gardenes et al., 2010),
    1 for phase-locked nodes and near 0 for independent ones.

    Parameters
    ----------
    phases : array_like
        Phases, shape (n_points, M).

    Returns
    -------
    np.ndarray
        Symmetric (M, M) matrix with ones on the diagonal.
    """
    phases = np.asarray(phases, dtype=float)
    z = np.exp(1j * phases)
    return np.abs(z.T @ np.conj(z)) / len(phases)


def global_order(data):
    """Synchronisation measure R(t) of Rubido et al.

    R(t) is the variance over time of the network average, divided by the
    network average of each node's variance over time, both computed from
    the first t points. It is 1 for fully synchronised nodes and near 1/M for
    independent ones.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, M).

    Returns
    -------
    np.ndarray
        Shape (n_points,). The first value is undefined (the variance of a
        single point, 0/0) and returned as NaN.
    """
    X = np.asarray(data, dtype=float)
    t = np.arange(1, len(X) + 1)
    x_avg = X.mean(axis=1)
    numerator = np.cumsum(x_avg**2) / t - (np.cumsum(x_avg) / t) ** 2
    node_var = np.cumsum(X**2, axis=0) / t[:, None] - (np.cumsum(X, axis=0) / t[:, None]) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / node_var.mean(axis=1)


def deviation_from_sync(x1, x2):
    """Running average distance between two oscillators.

        E_i = (1 / i) sum_{k=1}^{i} |x2_k - x1_k|,   i = 1, ..., n

    E goes to zero when the two oscillators synchronise completely.

    Parameters
    ----------
    x1, x2 : array_like
        Time series of the two oscillators, of equal length n.

    Returns
    -------
    np.ndarray
        Shape (n,): the average of |x2 - x1| over the first 1, 2, ..., n points.

    Notes
    -----
    The thesis MATLAB code (`deviation_from_sync.m`, used for Fig. 4.3(e))
    divided the running sum by the time t_i instead of the number of points,
    which gives the average only for a unit step starting at t = 0 (for a step
    h it is 1/h times too large), and dropped the last two points.
    """
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    if x1.shape != x2.shape:
        raise ValueError(f"x1 and x2 must have the same length, got {x1.size} and {x2.size}")
    return np.cumsum(np.abs(x2 - x1)) / np.arange(1, x1.size + 1)
