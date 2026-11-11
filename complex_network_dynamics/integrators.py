"""Integrators that evolve a system together with its tangent frame and return
the Lyapunov spectrum.

All integrators share the same names: `f` and `df` (vector field and
Jacobian), `y0` and `t0` (initial state and time), `h` (step size), `n_steps`
(number of recorded points), `n_transient` (steps discarded first),
`qr_interval` (steps between QR re-orthonormalisations) and `lyap_mode`
("none", "spectrum" or "history"). Time runs on through the transient: the
first recorded point is at t0 + n_transient * h.
"""
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

_LYAP_MODES = ("none", "spectrum", "history")


def _check_options(lyap_mode, df, n_steps, qr_interval):
    """Validate the options shared by the fixed-step integrators; return the number of QR steps."""
    if lyap_mode not in _LYAP_MODES:
        raise ValueError(f"Invalid lyap_mode '{lyap_mode}'. Expected one of {_LYAP_MODES}")
    if lyap_mode == "none":
        return 0
    if df is None:
        raise ValueError(f"Jacobian function `df` must be provided when lyap_mode='{lyap_mode}'.")
    n_qr = (n_steps - 1) // qr_interval
    if n_qr == 0:
        raise ValueError(
            f"n_steps ({n_steps}) is too small for qr_interval ({qr_interval}). "
            f"At least {qr_interval + 1} steps are required."
        )
    return n_qr


def _qr_step(Q, lyap_sum):
    """Re-orthonormalise the tangent frame Q and add the log stretching factors to lyap_sum."""
    Q, R = np.linalg.qr(Q)
    diag_r = np.diag(R)
    signs = np.sign(diag_r)
    signs[signs == 0] = 1.0
    lyap_sum += np.log(np.maximum(np.abs(diag_r), 1e-300))
    return Q * signs


@njit
def _apply_qr_update(Q, lyap_sum):
    """Numba version of `_qr_step`, used by the solver of `build_rk4_lyapunov`.

    Re-orthonormalises the tangent frame Q (shape (d, d)) by QR decomposition,
    flips columns so the R factor has a positive diagonal, and adds the log
    stretching factors to lyap_sum (shape (d,), updated in place).
    """
    Q_new, R = np.linalg.qr(Q)
    diag_r = np.diag(R)

    for i in range(len(diag_r)):
        if diag_r[i] < 0:
            Q_new[:, i] = -Q_new[:, i]

    abs_diag = np.maximum(np.abs(diag_r), 1e-300)
    lyap_sum += np.log(abs_diag)

    return np.ascontiguousarray(Q_new), lyap_sum


def _results(lyap_mode, t_vals, traj, lyap_sum, total_time, lyap_times, lyap_history):
    """Return the outputs selected by lyap_mode."""
    if lyap_mode == "none":
        return t_vals, traj
    if lyap_mode == "spectrum":
        return t_vals, traj, np.sort(lyap_sum / total_time)[::-1]
    return t_vals, traj, lyap_times, lyap_history


# ---------------------------------------------------------------------------
# Forward Euler
# ---------------------------------------------------------------------------

def euler_with_lyapunov(
    f,
    y0,
    t0,
    h,
    n_steps,
    df=None,
    lyap_mode="spectrum",
    n_transient=1000,
    qr_interval=10,
    **params,
):
    """Integrate dy/dt = f(t, y) by forward Euler, with the Lyapunov spectrum.

    The tangent frame is evolved alongside the state, Q -> (I + h J) Q, and
    re-orthonormalised by QR decomposition every `qr_interval` steps.

    Parameters
    ----------
    f : callable
        Vector field f(t, y, **params).
    y0 : array_like
        Initial state.
    t0 : float
        Initial time.
    h : float
        Fixed step size.
    n_steps : int
        Number of recorded points after the transient (n_steps - 1 steps).
    df : callable, optional
        Jacobian df(t, y, **params). Required unless lyap_mode="none".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to compute besides the trajectory. Default is "spectrum".
    n_transient : int, optional
        Steps discarded before recording. Default is 1000.
    qr_interval : int, optional
        Number of steps between QR re-orthonormalisations. Default is 10.
    **params
        Extra keyword arguments passed to `f` and `df`.

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_steps,), starting at t0 + n_transient * h.
    traj : np.ndarray
        Trajectory, shape (n_steps, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first, shape (d,).
    lyap_times, lyap_history : np.ndarray
        Only if lyap_mode="history": the time of each QR step and the running
        Lyapunov estimates at those times, shape (n_qr, d).

    Raises
    ------
    ValueError
        If `lyap_mode` is invalid, `df` is missing, or `n_steps` is too small
        for `qr_interval`.

    See Also
    --------
    rk4_with_lyapunov : Fourth-order Runge-Kutta, far more accurate for the same step.
    """
    n_qr = _check_options(lyap_mode, df, n_steps, qr_interval)

    y = np.asarray(y0, dtype=float).ravel().copy()
    d = len(y)
    t = float(t0)

    def eval_f(t_eval, y_eval):
        return np.asarray(f(t_eval, y_eval, **params), dtype=float).ravel()

    # Transient (state only)
    for _ in range(n_transient):
        y += h * eval_f(t, y)
        t += h

    traj = np.zeros((n_steps, d), dtype=float)
    t_vals = np.zeros(n_steps, dtype=float)
    traj[0] = y
    t_vals[0] = t

    if lyap_mode == "none":
        for step in range(1, n_steps):
            y += h * eval_f(t, y)
            t += h
            traj[step] = y
            t_vals[step] = t
        return t_vals, traj

    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)
    lyap_times = np.zeros(n_qr, dtype=float)
    lyap_history = np.zeros((n_qr, d), dtype=float)
    qr_idx = 0

    for step in range(1, n_steps):
        fn = eval_f(t, y)
        Jn = np.asarray(df(t, y, **params), dtype=float)

        y += h * fn
        t += h
        traj[step] = y
        t_vals[step] = t

        # Variational step: Q_next = (I + h J) Q
        Q += h * (Jn @ Q)

        if step % qr_interval == 0:
            Q = _qr_step(Q, lyap_sum)
            if lyap_mode == "history":
                lyap_history[qr_idx] = lyap_sum / (step * h)
                lyap_times[qr_idx] = t
                qr_idx += 1

    return _results(lyap_mode, t_vals, traj, lyap_sum, n_qr * qr_interval * h, lyap_times, lyap_history)


# ---------------------------------------------------------------------------
# Fourth-order Runge-Kutta
# ---------------------------------------------------------------------------

def rk4_with_lyapunov(
    f,
    y0,
    t0,
    h,
    n_steps,
    df=None,
    lyap_mode="spectrum",
    n_transient=1000,
    qr_interval=10,
    **params,
):
    """Integrate dy/dt = f(t, y) by fourth-order Runge-Kutta, with the Lyapunov spectrum.

    The state and the tangent frame are advanced together by RK4, and the
    frame is re-orthonormalised by QR decomposition every `qr_interval` steps.

    Parameters
    ----------
    f : callable
        Vector field f(t, y, **params).
    y0 : array_like
        Initial state.
    t0 : float
        Initial time.
    h : float
        Fixed step size.
    n_steps : int
        Number of recorded points after the transient (n_steps - 1 steps).
    df : callable, optional
        Jacobian df(t, y, **params). Required unless lyap_mode="none".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to compute besides the trajectory. Default is "spectrum".
    n_transient : int, optional
        Steps discarded before recording. Default is 1000.
    qr_interval : int, optional
        Number of steps between QR re-orthonormalisations. Default is 10.
    **params
        Extra keyword arguments passed to `f` and `df`.

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_steps,), starting at t0 + n_transient * h.
    traj : np.ndarray
        Trajectory, shape (n_steps, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first, shape (d,).
    lyap_times, lyap_history : np.ndarray
        Only if lyap_mode="history": the time of each QR step and the running
        Lyapunov estimates at those times, shape (n_qr, d).

    Raises
    ------
    ValueError
        If `lyap_mode` is invalid, `df` is missing, or `n_steps` is too small
        for `qr_interval`.

    See Also
    --------
    build_rk4_lyapunov : The same method compiled with Numba, about 10 times faster.
    """
    n_qr = _check_options(lyap_mode, df, n_steps, qr_interval)

    y = np.asarray(y0, dtype=float).ravel().copy()
    d = len(y)
    t = float(t0)

    def eval_f(t_eval, y_eval):
        return np.asarray(f(t_eval, y_eval, **params), dtype=float).ravel()

    def rk4_state_step(t, y):
        k1 = eval_f(t, y)
        k2 = eval_f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = eval_f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = eval_f(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Transient (state only)
    for _ in range(n_transient):
        y = rk4_state_step(t, y)
        t += h

    traj = np.zeros((n_steps, d), dtype=float)
    t_vals = np.zeros(n_steps, dtype=float)
    traj[0] = y
    t_vals[0] = t

    if lyap_mode == "none":
        for step in range(1, n_steps):
            y = rk4_state_step(t, y)
            t += h
            traj[step] = y
            t_vals[step] = t
        return t_vals, traj

    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)
    lyap_times = np.zeros(n_qr, dtype=float)
    lyap_history = np.zeros((n_qr, d), dtype=float)
    qr_idx = 0

    for step in range(1, n_steps):
        # Stage 1
        k1 = eval_f(t, y)
        J1 = np.asarray(df(t, y, **params), dtype=float)
        K1_Q = J1 @ Q

        # Stage 2
        y2 = y + 0.5 * h * k1
        t2 = t + 0.5 * h
        k2 = eval_f(t2, y2)
        J2 = np.asarray(df(t2, y2, **params), dtype=float)
        K2_Q = J2 @ (Q + 0.5 * h * K1_Q)

        # Stage 3
        y3 = y + 0.5 * h * k2
        k3 = eval_f(t2, y3)
        J3 = np.asarray(df(t2, y3, **params), dtype=float)
        K3_Q = J3 @ (Q + 0.5 * h * K2_Q)

        # Stage 4
        y4 = y + h * k3
        t4 = t + h
        k4 = eval_f(t4, y4)
        J4 = np.asarray(df(t4, y4, **params), dtype=float)
        K4_Q = J4 @ (Q + h * K3_Q)

        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + (h / 6.0) * (K1_Q + 2 * K2_Q + 2 * K3_Q + K4_Q)
        t += h
        traj[step] = y
        t_vals[step] = t

        if step % qr_interval == 0:
            Q = _qr_step(Q, lyap_sum)
            if lyap_mode == "history":
                lyap_history[qr_idx] = lyap_sum / (step * h)
                lyap_times[qr_idx] = t
                qr_idx += 1

    return _results(lyap_mode, t_vals, traj, lyap_sum, n_qr * qr_interval * h, lyap_times, lyap_history)


# ---------------------------------------------------------------------------
# Fourth-order Runge-Kutta, compiled with Numba
# ---------------------------------------------------------------------------

def build_rk4_lyapunov(f, df):
    """Build a Numba-compiled `rk4_with_lyapunov` for one system.

    `f` and `df` are compiled into the solver, so it runs entirely in machine
    code, about 10 times faster than `rk4_with_lyapunov`. The solver uses the
    same method, names and outputs as `rk4_with_lyapunov` and gives the same
    results, except that the model parameters are passed as a tuple `args`.

    Parameters
    ----------
    f : callable
        Numba-compiled vector field f(t, y, args), e.g. `lorenz_njit`.
    df : callable
        Numba-compiled Jacobian df(t, y, args), e.g. `lorenz_jacobian_njit`.

    Returns
    -------
    callable
        solver(y0, t0, h, n_steps, args, lyap_mode="spectrum", n_transient=1000,
        qr_interval=10), returning (t_vals, traj) for lyap_mode="none",
        (t_vals, traj, spectrum) for "spectrum" and
        (t_vals, traj, lyap_times, lyap_history) for "history", exactly as
        `rk4_with_lyapunov`. The first call compiles the solver, which takes
        a few seconds.

    See Also
    --------
    rk4_with_lyapunov : The same method in plain Python, with keyword parameters.

    Examples
    --------
    >>> solver = build_rk4_lyapunov(lorenz_njit, lorenz_jacobian_njit)
    >>> t, traj, spectrum = solver(np.ones(3), 0.0, 0.01, 50_000, (10.0, 28.0, 8 / 3))
    """

    @njit
    def rk4_core(y0, t0, h, n_steps, args, compute_lyap, n_transient, qr_interval):
        d = len(y0)
        y = np.copy(y0)
        t = t0

        # Transient (state only)
        for _ in range(n_transient):
            k1 = f(t, y, args)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1, args)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2, args)
            k4 = f(t + h, y + h * k3, args)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h

        traj = np.zeros((n_steps, d), dtype=np.float64)
        t_vals = np.zeros(n_steps, dtype=np.float64)
        traj[0] = y
        t_vals[0] = t

        n_qr = (n_steps - 1) // qr_interval if compute_lyap else 0
        Q = np.eye(d, dtype=np.float64)
        lyap_sum = np.zeros(d, dtype=np.float64)
        lyap_times = np.zeros(n_qr, dtype=np.float64)
        lyap_history = np.zeros((n_qr, d), dtype=np.float64)
        qr_idx = 0

        for step in range(1, n_steps):
            k1 = f(t, y, args)
            y2 = y + 0.5 * h * k1
            t2 = t + 0.5 * h
            k2 = f(t2, y2, args)
            y3 = y + 0.5 * h * k2
            k3 = f(t2, y3, args)
            y4 = y + h * k3
            t4 = t + h
            k4 = f(t4, y4, args)

            if compute_lyap:
                K1_Q = df(t, y, args) @ Q
                K2_Q = df(t2, y2, args) @ (Q + 0.5 * h * K1_Q)
                K3_Q = df(t2, y3, args) @ (Q + 0.5 * h * K2_Q)
                K4_Q = df(t4, y4, args) @ (Q + h * K3_Q)
                Q = Q + (h / 6.0) * (K1_Q + 2.0 * K2_Q + 2.0 * K3_Q + K4_Q)

            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            traj[step] = y
            t_vals[step] = t

            if compute_lyap and step % qr_interval == 0:
                Q, lyap_sum = _apply_qr_update(Q, lyap_sum)
                lyap_history[qr_idx, :] = lyap_sum / (step * h)
                lyap_times[qr_idx] = t
                qr_idx += 1

        return t_vals, traj, lyap_sum, lyap_times, lyap_history

    def solver(y0, t0, h, n_steps, args, lyap_mode="spectrum", n_transient=1000, qr_interval=10):
        n_qr = _check_options(lyap_mode, df, n_steps, qr_interval)
        t_vals, traj, lyap_sum, lyap_times, lyap_history = rk4_core(
            np.asarray(y0, dtype=np.float64).ravel().copy(), float(t0), float(h), int(n_steps), args,
            lyap_mode != "none", int(n_transient), int(qr_interval),
        )
        return _results(lyap_mode, t_vals, traj, lyap_sum, n_qr * qr_interval * h, lyap_times, lyap_history)

    return solver


# ---------------------------------------------------------------------------
# Adaptive SciPy solvers
# ---------------------------------------------------------------------------

def solve_ivp_with_lyapunov(
    f,
    y0,
    t0,
    t_total,
    df=None,
    lyap_mode="spectrum",
    n_transient=200,
    dt_qr=0.05,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
    **params,
):
    """Integrate dy/dt = f(t, y) by an adaptive SciPy solver, with the Lyapunov spectrum.

    The state and the tangent frame are integrated together by
    `scipy.integrate.solve_ivp` over windows of length `dt_qr`; after each
    window the frame is re-orthonormalised by QR decomposition and the state
    is recorded.

    Parameters
    ----------
    f : callable
        Vector field f(t, y, **params).
    y0 : array_like
        Initial state.
    t0 : float
        Initial time.
    t_total : float
        Integration time recorded after the transient.
    df : callable, optional
        Jacobian df(t, y, **params). Required unless lyap_mode="none".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to compute besides the trajectory. Default is "spectrum".
    n_transient : int, optional
        Windows of length `dt_qr` discarded before recording. Default is 200.
    dt_qr : float, optional
        Length of each window: the time between QR re-orthonormalisations
        and between recorded points. Default is 0.05.
    method : str, optional
        Any `scipy.integrate.solve_ivp` method. Default is "RK45".
    rtol, atol : float, optional
        Solver tolerances. Defaults are 1e-7 and 1e-9.
    **params
        Extra keyword arguments passed to `f` and `df` (e.g. A, K, sigma).

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_windows + 1,), starting at t0 + n_transient * dt_qr,
        where n_windows = round(t_total / dt_qr).
    traj : np.ndarray
        Trajectory sampled every `dt_qr`, shape (n_windows + 1, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first, shape (d,).
    lyap_times, lyap_history : np.ndarray
        Only if lyap_mode="history": the time of each QR step and the running
        Lyapunov estimates at those times, shape (n_windows, d).

    Raises
    ------
    ValueError
        If `lyap_mode` is invalid or `df` is missing.

    See Also
    --------
    rk4_with_lyapunov : Fixed-step RK4, faster for smooth systems.
    """
    _check_options(lyap_mode, df, 2, 1)

    y = np.asarray(y0, dtype=float).ravel().copy()
    d = len(y)
    t = float(t0)

    def f_eval(t_eval, y_eval):
        return np.asarray(f(t_eval, y_eval, **params), dtype=float).ravel()

    def augmented_rhs(t_eval, y_aug):
        state = y_aug[:d]
        Q_mat = y_aug[d:].reshape(d, d)
        J = np.asarray(df(t_eval, state, **params), dtype=float)
        return np.concatenate([f_eval(t_eval, state), (J @ Q_mat).ravel()])

    # Transient (state only)
    t_transient = n_transient * dt_qr
    if t_transient > 0:
        sol = solve_ivp(f_eval, (t, t + t_transient), y, method=method, rtol=rtol, atol=atol)
        y = sol.y[:, -1]
        t = sol.t[-1]

    n_windows = max(1, int(round(t_total / dt_qr)))
    traj = np.zeros((n_windows + 1, d), dtype=float)
    t_vals = np.zeros(n_windows + 1, dtype=float)
    traj[0] = y
    t_vals[0] = t
    t_start = t

    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)
    lyap_times = np.zeros(n_windows, dtype=float)
    lyap_history = np.zeros((n_windows, d), dtype=float)

    for step in range(n_windows):
        t_end = t_start + (step + 1) * dt_qr
        if lyap_mode == "none":
            sol = solve_ivp(f_eval, (t, t_end), y, method=method, rtol=rtol, atol=atol)
            y = sol.y[:, -1]
        else:
            sol = solve_ivp(augmented_rhs, (t, t_end), np.concatenate([y, Q.ravel()]),
                            method=method, rtol=rtol, atol=atol)
            y = sol.y[:d, -1]
            Q = _qr_step(sol.y[d:, -1].reshape(d, d), lyap_sum)
            lyap_history[step] = lyap_sum / (t_end - t_start)
            lyap_times[step] = t_end
        t = t_end
        traj[step + 1] = y
        t_vals[step + 1] = t

    return _results(lyap_mode, t_vals, traj, lyap_sum, n_windows * dt_qr, lyap_times, lyap_history)
