import numpy as np
from scipy.signal import hilbert

# Phases and time series are (n_points, n_nodes) arrays, one column per node.


def hilbert_phases(data):
    """Instantaneous phases from the Hilbert transform.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes).

    Returns
    -------
    np.ndarray
        Unwrapped phase of each node's analytic signal, same shape as `data`.
    """
    data = np.asarray(data, dtype=float)
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


def deviation_from_sync(x1, x2, tvalues):
    """Average distance between two oscillators over time.

    E(t_i) = (1 / t_i) * sum_{k <= i} |x2(t_k) - x1(t_k)|, for i = 2, ..., n - 2
    (1-based).

    Parameters
    ----------
    x1, x2 : array_like
        Time series of the two oscillators.
    tvalues : array_like
        Time of each point.

    Returns
    -------
    np.ndarray
        Shape (len(tvalues) - 3,).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    tvalues = np.asarray(tvalues, dtype=float)
    n = len(tvalues) - 2
    running = np.cumsum(np.abs(x2[:n] - x1[:n]))
    return running[1:] / tvalues[1:n]
