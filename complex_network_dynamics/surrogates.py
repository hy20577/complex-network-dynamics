import numpy as np
from scipy.signal import hilbert

# All functions take time series as an (n_points, n_nodes) array, one column per
# node, and return surrogates as an (n_surrogates, n_points, n_nodes) array.


def phaseran(data, n_surrogates=1, rng=None):
    """Phase-randomised (twin) surrogates that keep the cross-correlations.

    The same random phases are added to every column's Fourier transform
    (Prichard & Theiler, 1994), so each surrogate keeps the power spectrum of
    every node and the linear correlations between nodes, but destroys any
    nonlinear structure.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes). If n_points is even, the last
        sample is dropped, as in the MATLAB code.
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
    data = np.asarray(data, dtype=float)
    n_frames, n_nodes = data.shape
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
    """Amplitude-phase modulated surrogate data (APMSD), MATLAB `HT_mix_A_theta`.

    Each node's analytic signal A(t) exp(i theta(t)) is computed with the
    Hilbert transform. In every surrogate, a random fraction `pc_amplitude` of
    the time points get new amplitudes drawn uniformly between the smallest
    and largest amplitude in the data, and a separate random fraction
    `pc_phase` get new phases drawn uniformly between the smallest and largest
    phase. The surrogate is the real part of the modified signal. Randomising
    amplitudes removes amplitude correlation between nodes; randomising phases
    removes phase synchronisation.

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
    """
    rng = np.random.default_rng(rng)
    data = np.asarray(data, dtype=float)
    n_points, n_nodes = data.shape

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

    MATLAB `Mir_surrogate_data` option "randsd".

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
    n_points, n_nodes = np.shape(data)
    return rng.random((n_surrogates, n_points, n_nodes))


def correlated_random_surrogates(data, n_surrogates=1, correlation=0.5, n_points=1000, rng=None):
    """Gaussian surrogates with the same correlation between every pair of nodes.

    MATLAB `Mir_surrogate_data` option "corr_randsd": standard normal samples
    multiplied by the Cholesky factor of a matrix with ones on the diagonal and
    `correlation` elsewhere. As in the MATLAB code, the length is fixed
    (`n_points`) rather than taken from the data.

    Parameters
    ----------
    data : array_like
        Time series, shape (n_points, n_nodes); only the number of nodes is used.
    n_surrogates : int, optional
        Number of surrogate datasets. Default is 1.
    correlation : float, optional
        Correlation between every pair of nodes. Default is 0.5.
    n_points : int, optional
        Length of each surrogate. Default is 1000.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    np.ndarray
        Surrogates, shape (n_surrogates, n_points, n_nodes).
    """
    rng = np.random.default_rng(rng)
    n_nodes = np.shape(data)[1]
    R = np.full((n_nodes, n_nodes), correlation)
    np.fill_diagonal(R, 1.0)
    upper = np.linalg.cholesky(R).T  # MATLAB chol() returns the upper factor
    return rng.standard_normal((n_surrogates, n_points, n_nodes)) @ upper
