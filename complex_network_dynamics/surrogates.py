import numpy as np
from scipy.signal import hilbert

# All functions take time series as an (n_points, n_nodes) array, one column per
# node, and return surrogates as an (n_surrogates, n_points, n_nodes) array.


def _check_data(data):
    """Return the time series as a float (n_points, n_nodes) array."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must have shape (n_points, n_nodes), one column per node; got shape {data.shape}. "
            "For a single time series x, pass x[:, None]."
        )
    return data


def phaseran(data, n_surrogates=1, rng=None):
    """Phase-randomised surrogates that keep the cross-correlations.

    The same random phases are added to every column's Fourier transform
    (Prichard & Theiler, 1994), so each surrogate keeps the power spectrum of
    every node and the linear correlations between nodes, but destroys any
    nonlinear structure. The thesis code calls this surrogate type `twnsd`.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes). If n_points is even, the last
        sample is dropped.
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 1.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    np.ndarray
        Surrogates, shape (n_surrogates, n_frames, n_nodes), where n_frames is
        n_points rounded down to an odd number.
    """
    rng = np.random.default_rng(rng)
    data = _check_data(data)
    n_frames, n_nodes = data.shape
    if n_frames < 3:
        raise ValueError(f"data needs at least 3 time points to randomise its Fourier phases; got {n_frames}.")
    if n_frames % 2 == 0:
        n_frames -= 1
        data = data[:n_frames]

    half = (n_frames - 1) // 2
    spectrum = np.fft.fft(data, axis=0)

    surrogates = np.empty((n_surrogates, n_frames, n_nodes))
    for k in range(n_surrogates):
        phases = np.exp(2j * np.pi * rng.random(half))[:, None]
        randomised = spectrum.copy()
        randomised[1:half + 1] *= phases
        randomised[half + 1:] *= np.conj(phases[::-1])
        surrogates[k] = np.real(np.fft.ifft(randomised, axis=0))
    return surrogates


def amplitude_phase_surrogates(data, n_surrogates=1, pc_amplitude=1.0, pc_phase=0.5, rng=None):
    """Amplitude-phase modulated surrogate data (APMSD).

    Each node's analytic signal A(t) exp(i theta(t)) is computed with the
    Hilbert transform. In every surrogate, a random fraction `pc_amplitude` of
    the time points get new amplitudes drawn uniformly between the smallest
    and largest amplitude in the data, and a separate random fraction
    `pc_phase` get new phases drawn uniformly between the smallest and largest
    phase. The surrogate is the real part of the modified signal. Randomising
    amplitudes removes amplitude correlation between nodes; randomising phases
    removes phase synchronisation.

    As in the thesis code (`HT_mix_A_theta`), the randomised time points are
    the same for all nodes, while every node gets its own new values; the
    smallest and largest amplitude and phase are taken over all nodes
    together; and the Hilbert transform is applied to the data as given, with
    the mean left in (unlike `hilbert_phases`, which removes it by default).

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes).
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 1.
    pc_amplitude : float, optional
        Fraction of time points whose amplitudes are randomised, in [0, 1].
        Default is 1.
    pc_phase : float, optional
        Fraction of time points whose phases are randomised, in [0, 1].
        Default is 0.5.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    np.ndarray
        Surrogates, shape (n_surrogates, n_points, n_nodes).

    Raises
    ------
    ValueError
        If `data` is not two-dimensional, or a fraction is outside [0, 1].
    """
    rng = np.random.default_rng(rng)
    data = _check_data(data)
    n_points, n_nodes = data.shape
    for name, pc in (("pc_amplitude", pc_amplitude), ("pc_phase", pc_phase)):
        if not 0 <= pc <= 1:
            raise ValueError(f"{name} is a fraction of the time points and must be between 0 and 1; got {pc}.")

    analytic = hilbert(data, axis=0)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    amp_min, amp_max = amplitude.min(), amplitude.max()
    ph_min, ph_max = phase.min(), phase.max()
    n_amp = int(np.ceil(pc_amplitude * n_points))
    n_ph = int(np.ceil(pc_phase * n_points))

    surrogates = np.empty((n_surrogates, n_points, n_nodes))
    for k in range(n_surrogates):
        new_amplitude = amplitude.copy()
        rows = rng.permutation(n_points)[:n_amp]
        new_amplitude[rows] = rng.random((n_amp, n_nodes)) * (amp_max - amp_min) + amp_min

        new_phase = phase.copy()
        rows = rng.permutation(n_points)[:n_ph]
        new_phase[rows] = rng.random((n_ph, n_nodes)) * (ph_max - ph_min) + ph_min

        surrogates[k] = np.real(new_amplitude * np.exp(1j * new_phase))
    return surrogates


def random_surrogates(data, n_surrogates=1, rng=None):
    """Uniform random surrogates in [0, 1) with the same shape as the data.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes); only its shape is used.
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 1.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    np.ndarray
        Surrogates, shape (n_surrogates, n_points, n_nodes).
    """
    rng = np.random.default_rng(rng)
    n_points, n_nodes = _check_data(data).shape
    return rng.random((n_surrogates, n_points, n_nodes))


def correlated_random_surrogates(data, n_surrogates=1, correlation=0.5, n_points=None, rng=None):
    """Gaussian surrogates with the same correlation between every pair of nodes.

    Standard normal samples multiplied by the Cholesky factor of a matrix with
    ones on the diagonal and `correlation` elsewhere. The thesis code
    (`corr_randsd`) always made 1000 points; here the surrogates are as long
    as the data unless `n_points` is given.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes); only its shape is used.
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 1.
    correlation : float, optional
        Correlation between every pair of nodes, above -1/(n_nodes - 1) and
        below 1. Default is 0.5.
    n_points : int, optional
        Length of each surrogate. Default is the length of the data.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    np.ndarray
        Surrogates, shape (n_surrogates, n_points, n_nodes).

    Raises
    ------
    ValueError
        If `data` is not two-dimensional, or no set of nodes can have the
        requested correlation between every pair.
    """
    rng = np.random.default_rng(rng)
    data_points, n_nodes = _check_data(data).shape
    n_points = data_points if n_points is None else int(n_points)
    lowest = -1.0 / (n_nodes - 1) if n_nodes > 1 else -1.0
    if not lowest < correlation < 1:
        raise ValueError(
            f"correlation must be above {lowest:.3g} (= -1/(n_nodes - 1)) and below 1 for {n_nodes} nodes; "
            f"got {correlation}."
        )
    R = np.full((n_nodes, n_nodes), correlation)
    np.fill_diagonal(R, 1.0)
    upper = np.linalg.cholesky(R).T  # Upper Cholesky factor
    return rng.standard_normal((n_surrogates, n_points, n_nodes)) @ upper
