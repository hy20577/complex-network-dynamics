import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


#### ### 
@njit
def _apply_qr_update(Q, lyap_sum):
    """Re-orthonormalize the tangent frame and accumulate Lyapunov sums.

    Internal helper (Numba-compiled) used by the fast RK4 integrators.

    Parameters
    ----------
    Q : np.ndarray
        Evolved tangent frame, shape (d, d).
    lyap_sum : np.ndarray
        Running sum of log stretching factors, shape (d,). Updated in place.

    Returns
    -------
    Q_new : np.ndarray
        Orthonormal frame with a positive-diagonal R factor.
    lyap_sum : np.ndarray
        Updated running sum.
    """
    Q_new, R = np.linalg.qr(Q)
    diag_r = np.diag(R)
    
    for i in range(len(diag_r)):
        if diag_r[i] < 0:
            Q_new[:, i] = -Q_new[:, i]
            
    abs_diag = np.maximum(np.abs(diag_r), 1e-300)
    lyap_sum += np.log(abs_diag)
    
    return np.ascontiguousarray(Q_new), lyap_sum


def build_rk4_lyapunov(f, df):
    """Build a Numba-compiled RK4 + Lyapunov solver for a specific system.

    `f` and `df` are baked into the compiled solver, so it runs entirely in
    machine code. Unlike `rk4_lyapunov_fast`, the returned solver also runs a
    transient burn-in.

    Parameters
    ----------
    f : callable
        Numba-compiled vector field f(t, y, args), e.g. `lorenz_njit`.
    df : callable
        Numba-compiled Jacobian df(t, y, args), e.g. `lorenz_jacobian_njit`.

    Returns
    -------
    callable
        solver(y0, t0, h, n_steps, qr_interval, args, n_transient=1000,
        save_history=False), returning
        (t_vals, traj, final_spectrum, lyap_times, lyap_history).

    Examples
    --------
    >>> solver = build_rk4_lyapunov(lorenz_njit, lorenz_jacobian_njit)
    >>> t, traj, spec, _, _ = solver(np.ones(3), 0.0, 0.01, 50_000, 10, (10.0, 28.0, 8/3))
    """

    # Remove fastmath=True here as well
    @njit
    def custom_rk4_solver(y0, t0, h, n_steps, qr_interval, args, n_transient=1000, save_history=False):
        d = len(y0)
        y = np.copy(y0)
        t = float(t0)

        # ---------------------------------------------------------
        # 1. TRANSIENT BURN-IN (State only, no Jacobian)
        # ---------------------------------------------------------
        for _ in range(n_transient):
            k1 = f(t, y, args)
            y2 = y + 0.5 * h * k1
            k2 = f(t + 0.5 * h, y2, args)
            y3 = y + 0.5 * h * k2
            k3 = f(t + 0.5 * h, y3, args)
            y4 = y + h * k3
            k4 = f(t + h, y4, args)
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            
        # Reset time back to t0 (or keep it continuous, based on preference)
        t = float(t0)

        # ---------------------------------------------------------
        # 2. MAIN LYAPUNOV LOOP
        # ---------------------------------------------------------
        traj = np.zeros((n_steps, d), dtype=np.float64)
        t_vals = np.zeros(n_steps, dtype=np.float64)

        Q = np.eye(d, dtype=np.float64)
        lyap_sum = np.zeros(d, dtype=np.float64)

        n_qr_events = n_steps // qr_interval
        if save_history:
            lyap_history = np.zeros((n_qr_events, d), dtype=np.float64)
            lyap_times = np.zeros(n_qr_events, dtype=np.float64)
        else:
            lyap_history = np.zeros((0, d), dtype=np.float64)
            lyap_times = np.zeros(0, dtype=np.float64)
            
        qr_idx = 0

        for step in range(n_steps):
            traj[step] = y
            t_vals[step] = t

            k1 = f(t, y, args)
            J1 = df(t, y, args)
            K1_Q = J1 @ Q

            y2 = y + 0.5 * h * k1
            t2 = t + 0.5 * h
            k2 = f(t2, y2, args)
            J2 = df(t2, y2, args)
            K2_Q = J2 @ (Q + 0.5 * h * K1_Q)

            y3 = y + 0.5 * h * k2
            t3 = t + 0.5 * h
            k3 = f(t3, y3, args)
            J3 = df(t3, y3, args)
            K3_Q = J3 @ (Q + 0.5 * h * K2_Q)

            y4 = y + h * k3
            t4 = t + h
            k4 = f(t4, y4, args)
            J4 = df(t4, y4, args)
            K4_Q = J4 @ (Q + h * K3_Q)

            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            Q += (h / 6.0) * (K1_Q + 2.0 * K2_Q + 2.0 * K3_Q + K4_Q)
            t += h

            if step > 0 and step % qr_interval == 0:
                Q, lyap_sum = _apply_qr_update(Q, lyap_sum)
                if save_history:
                    elapsed_time = step * h
                    lyap_history[qr_idx, :] = lyap_sum / elapsed_time
                    lyap_times[qr_idx] = t
                    qr_idx += 1

        total_lyap_time = (n_steps // qr_interval * qr_interval) * h
        final_spectrum = np.sort(lyap_sum / total_lyap_time)[::-1]

        return t_vals, traj, final_spectrum, lyap_times, lyap_history
    return custom_rk4_solver

@njit
def rk4_lyapunov_fast(f, df, y0, t0, h, n_steps, qr_interval, args, save_history=False):
    """
    Fixed-step RK4 with Lyapunov spectrum and optional history tracking.

    Numba-compiled; `f` and `df` must also be Numba-compiled (e.g.
    `lorenz_njit`). No transient is discarded.

    Parameters
    ----------
    f : callable
        Vector field f(t, y, args).
    df : callable
        Jacobian df(t, y, args).
    y0 : np.ndarray
        Initial state (float64).
    t0 : float
        Initial time.
    h : float
        Step size.
    n_steps : int
        Number of steps.
    qr_interval : int
        Steps between QR re-orthonormalizations.
    args : tuple
        Parameters passed to `f` and `df`.
    save_history : bool, optional
        Record the running spectrum at each QR step. Default is False.

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_steps,).
    traj : np.ndarray
        Trajectory, shape (n_steps, d).
    final_spectrum : np.ndarray
        Lyapunov exponents, largest first, shape (d,).
    lyap_times : np.ndarray
        Times of each QR step (empty if save_history is False).
    lyap_history : np.ndarray
        Running Lyapunov estimates at each QR step, shape (n_qr, d)
        (empty if save_history is False).
    """
    d = len(y0)
    y = np.copy(y0)
    t = float(t0)

    # Pre-allocate trajectory arrays
    traj = np.zeros((n_steps, d), dtype=np.float64)
    t_vals = np.zeros(n_steps, dtype=np.float64)

    # Initialize orthonormal tangent frame Q
    Q = np.eye(d, dtype=np.float64)
    lyap_sum = np.zeros(d, dtype=np.float64)

    # Numba-compliant dynamic array allocation for history tracking
    n_qr_events = n_steps // qr_interval
    if save_history:
        lyap_history = np.zeros((n_qr_events, d), dtype=np.float64)
        lyap_times = np.zeros(n_qr_events, dtype=np.float64)
    else:
        # Empty arrays maintain consistent return typing when history is disabled
        lyap_history = np.zeros((0, d), dtype=np.float64)
        lyap_times = np.zeros(0, dtype=np.float64)
        
    qr_idx = 0

    # Main RK4 Loop
    for step in range(n_steps):
        traj[step] = y
        t_vals[step] = t

        # Stage 1
        k1 = f(t, y, args)
        J1 = df(t, y, args)
        K1_Q = J1 @ Q

        # Stage 2
        y2 = y + 0.5 * h * k1
        t2 = t + 0.5 * h
        k2 = f(t2, y2, args)
        J2 = df(t2, y2, args)
        K2_Q = J2 @ (Q + 0.5 * h * K1_Q)

        # Stage 3
        y3 = y + 0.5 * h * k2
        t3 = t + 0.5 * h
        k3 = f(t3, y3, args)
        J3 = df(t3, y3, args)
        K3_Q = J3 @ (Q + 0.5 * h * K2_Q)

        # Stage 4
        y4 = y + h * k3
        t4 = t + h
        k4 = f(t4, y4, args)
        J4 = df(t4, y4, args)
        K4_Q = J4 @ (Q + h * K3_Q)

        # Update state and tangent frame
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        Q += (h / 6.0) * (K1_Q + 2.0 * K2_Q + 2.0 * K3_Q + K4_Q)
        t += h

        # Apply QR update
        if step > 0 and step % qr_interval == 0:
            Q, lyap_sum = _apply_qr_update(Q, lyap_sum)
            
            # Record history if toggled
            if save_history:
                elapsed_time = step * h
                # Record the running average (spectrum over time)
                lyap_history[qr_idx, :] = lyap_sum / elapsed_time
                lyap_times[qr_idx] = t
                qr_idx += 1

    # Compute final spectrum
    total_lyap_time = (n_steps // qr_interval * qr_interval) * h
    final_spectrum = np.sort(lyap_sum / total_lyap_time)[::-1]

    # Always returns 5 variables. If save_history=False, the last two are empty.
    return t_vals, traj, final_spectrum, lyap_times, lyap_history
# ## # # # # #### ##  # ##### # # # # # # #  EULER (forward) with Lyapunov  # ### ## # ## ## ## ### ### # # # ## #                


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
    """Integrates dy/dt = f(t, y) via Forward Euler with flexible Lyapunov options.

    The tangent frame is evolved alongside the state and re-orthonormalized
    by QR decomposition every `qr_interval` steps.

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
        Number of recorded steps (after the transient).
    df : callable, optional
        Jacobian df(t, y, **params). Required unless lyap_mode="none".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to compute besides the trajectory. Default is "spectrum".
    n_transient : int, optional
        Burn-in steps discarded before recording. Default is 1000.
    qr_interval : int, optional
        Number of steps between QR re-orthonormalizations. Default is 10.
    **params
        Extra keyword arguments passed to `f` and `df`.

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_steps,).
    traj : np.ndarray
        Trajectory, shape (n_steps, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first, shape (d,).
    lyap_t, lyap_history : np.ndarray
        Only if lyap_mode="history": times of each QR step and the running
        Lyapunov estimates at those times, shape (n_qr, d).

    Raises
    ------
    ValueError
        If `lyap_mode` is invalid, `df` is missing, or `n_steps` is too small
        for `qr_interval`.
    """
    valid_modes = {"none", "spectrum", "history"}
    if lyap_mode not in valid_modes:
        raise ValueError(
            f"Invalid lyap_mode '{lyap_mode}'. Expected one of {valid_modes}"
        )

    if lyap_mode in {"spectrum", "history"} and df is None:
        raise ValueError(
            f"Jacobian function `df` must be provided when lyap_mode='{lyap_mode}'."
        )

    # 1. Force strictly 1D float vector
    y = np.asarray(y0, dtype=float).ravel().copy()
    d = len(y)
    t = float(t0)

    # 2. Transient burn-in (state only)
    for _ in range(n_transient):
        fn = np.asarray(f(t, y, **params), dtype=float).ravel()
        y += h * fn
        t += h

    # 3. Allocate trajectory storage
    traj = np.zeros((n_steps, d), dtype=float)
    t_vals = np.zeros(n_steps, dtype=float)
    traj[0] = y
    t_vals[0] = t

    # Fast path: No Lyapunov calculation
    if lyap_mode == "none":
        for step in range(1, n_steps):
            fn = np.asarray(f(t, y, **params), dtype=float).ravel()
            y += h * fn
            t += h
            traj[step] = y
            t_vals[step] = t
        return t_vals, traj

    # 4. Setup for Lyapunov evaluation
    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)

    n_qr_events = (n_steps - 1) // qr_interval
    if n_qr_events == 0:
        raise ValueError(
            f"n_steps ({n_steps}) is too small for qr_interval ({qr_interval}). "
            f"At least {qr_interval + 1} steps are required."
        )

    if lyap_mode == "history":
        lyap_history = np.zeros((n_qr_events, d), dtype=float)
        lyap_t = np.zeros(n_qr_events, dtype=float)
        qr_idx = 0

    # 5. Main simulation loop with direct tangent frame evolution
    for step in range(1, n_steps):
        fn = np.asarray(f(t, y, **params), dtype=float).ravel()
        Jn = np.asarray(df(t, y, **params), dtype=float)

        # Euler state step
        y += h * fn
        t += h
        traj[step] = y
        t_vals[step] = t

        # Evolve variational frame Q directly: Q_next = (I + h * J) @ Q = Q + h * (J @ Q)
        Q += h * (Jn @ Q)

        # Re-orthonormalize every qr_interval steps
        if step % qr_interval == 0:
            Q, R = np.linalg.qr(Q)

            # Preserve frame orientation consistency
            diag_r = np.diag(R)
            signs = np.sign(diag_r)
            signs[signs == 0] = 1.0
            Q *= signs
            abs_diag = np.maximum(np.abs(diag_r), 1e-300)

            lyap_sum += np.log(abs_diag)

            if lyap_mode == "history":
                elapsed_time = step * h
                lyap_history[qr_idx] = lyap_sum / elapsed_time
                lyap_t[qr_idx] = t
                qr_idx += 1

    # 6. Final spectrum computation
    total_lyap_time = (n_qr_events * qr_interval) * h
    final_spectrum = np.sort(lyap_sum / total_lyap_time)[::-1]

    if lyap_mode == "spectrum":
        return t_vals, traj, final_spectrum
    else:
        return t_vals, traj, lyap_t, lyap_history
               
# ## # # # # #### ##  # ##### # # # # # # #  RK4 Integrator with Lyapunov  # ### ## # ## ## ## ### ### # # # ## #                

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
    """
    Integrates dy/dt = f(t, y) via 4th-order Runge-Kutta (RK4) with Lyapunov spectrum.

    The state and the tangent frame are advanced together by RK4 and the
    frame is re-orthonormalized by QR decomposition every `qr_interval` steps.

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
        Number of recorded steps (after the transient).
    df : callable, optional
        Jacobian df(t, y, **params). Required unless lyap_mode="none".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to compute besides the trajectory. Default is "spectrum".
    n_transient : int, optional
        Burn-in steps discarded before recording. Default is 1000.
    qr_interval : int, optional
        Number of steps between QR re-orthonormalizations. Default is 10.
    **params
        Extra keyword arguments passed to `f` and `df`.

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_steps,).
    traj : np.ndarray
        Trajectory, shape (n_steps, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first, shape (d,).
    lyap_t, lyap_history : np.ndarray
        Only if lyap_mode="history": times of each QR step and the running
        Lyapunov estimates at those times, shape (n_qr, d).

    Raises
    ------
    ValueError
        If `lyap_mode` is invalid, `df` is missing, or `n_steps` is too small
        for `qr_interval`.
    """
    valid_modes = {"none", "spectrum", "history"}
    if lyap_mode not in valid_modes:
        raise ValueError(
            f"Invalid lyap_mode '{lyap_mode}'. Expected one of {valid_modes}"
        )

    if lyap_mode in {"spectrum", "history"} and df is None:
        raise ValueError(
            f"Jacobian function `df` must be provided when lyap_mode='{lyap_mode}'."
        )

    # 1. Guarantee y is strictly a 1D float array
    y = np.asarray(y0, dtype=float).ravel().copy()
    d = len(y)
    t = float(t0)

    # Helper to enforce flat 1D output from f
    def eval_f(t_eval, y_eval):
        return np.asarray(f(t_eval, y_eval, **params), dtype=float).ravel()

    # 2. Transient burn-in via standard RK4 (state only)
    for _ in range(n_transient):
        k1 = eval_f(t, y)
        k2 = eval_f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = eval_f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = eval_f(t + h, y + h * k3)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h

    # 3. Trajectory storage
    traj = np.zeros((n_steps, d), dtype=float)
    t_vals = np.zeros(n_steps, dtype=float)
    traj[0] = y
    t_vals[0] = t

    # Fast path: No Lyapunov computation
    if lyap_mode == "none":
        for step in range(1, n_steps):
            k1 = eval_f(t, y)
            k2 = eval_f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = eval_f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = eval_f(t + h, y + h * k3)
            y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            traj[step] = y
            t_vals[step] = t
        return t_vals, traj

    # 4. Setup for Lyapunov evaluation
    n_qr_events = (n_steps - 1) // qr_interval
    if n_qr_events == 0:
        raise ValueError(
            f"n_steps ({n_steps}) is too small for qr_interval ({qr_interval}). "
            f"At least {qr_interval + 1} steps are required."
        )

    # Initialize orthonormal tangent frame Q directly
    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)

    if lyap_mode == "history":
        lyap_history = np.zeros((n_qr_events, d), dtype=float)
        lyap_t = np.zeros(n_qr_events, dtype=float)
        qr_idx = 0

    # 5. Main RK4 Loop: simultaneously integrate state and tangent frame Q
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
        t3 = t + 0.5 * h
        k3 = eval_f(t3, y3)
        J3 = np.asarray(df(t3, y3, **params), dtype=float)
        K3_Q = J3 @ (Q + 0.5 * h * K2_Q)

        # Stage 4
        y4 = y + h * k3
        t4 = t + h
        k4 = eval_f(t4, y4)
        J4 = np.asarray(df(t4, y4, **params), dtype=float)
        K4_Q = J4 @ (Q + h * K3_Q)

        # Update state
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        traj[step] = y
        t_vals[step] = t

        # Update tangent frame directly
        Q += (h / 6.0) * (K1_Q + 2 * K2_Q + 2 * K3_Q + K4_Q)

        # Re-orthonormalize every qr_interval steps
        if step % qr_interval == 0:
            Q, R = np.linalg.qr(Q)

            # Preserve frame orientation consistency
            diag_r = np.diag(R)
            signs = np.sign(diag_r)
            signs[signs == 0] = 1.0
            Q *= signs
            abs_diag = np.maximum(np.abs(diag_r), 1e-300)

            lyap_sum += np.log(abs_diag)

            if lyap_mode == "history":
                elapsed_time = step * h
                lyap_history[qr_idx] = lyap_sum / elapsed_time
                lyap_t[qr_idx] = t
                qr_idx += 1

    # 6. Compute final spectrum
    total_lyap_time = (n_qr_events * qr_interval) * h
    final_spectrum = np.sort(lyap_sum / total_lyap_time)[::-1]

    if lyap_mode == "spectrum":
        return t_vals, traj, final_spectrum
    else:
        return t_vals, traj, lyap_t, lyap_history
                
######    ######### ######## ####### ##### ADAPTIVE METHODS with Lyapunov ##### ##### ###### ##### 

import numpy as np
from scipy.integrate import solve_ivp

def solve_ivp_with_lyapunov(
    f,
    df,
    y0,
    t_total,
    dt_qr=0.05,
    method="RK45",
    lyap_mode="spectrum",
    n_transient_steps=200,
    rtol=1e-7,
    atol=1e-9,
    **params,  # Accepts A, K, sigma, rho, beta, etc.
):
    """
    Integrates system and tangent frame using SciPy's solve_ivp with periodic QR re-orthonormalization.

    Uses an adaptive SciPy solver inside each window of length `dt_qr`.

    Parameters
    ----------
    f : callable
        Vector field f(t, y, **params).
    df : callable
        Jacobian df(t, y, **params).
    y0 : array_like
        Initial state.
    t_total : float
        Total integration time after the transient.
    dt_qr : float, optional
        Time between QR re-orthonormalizations (and between saved points).
        Default is 0.05.
    method : str, optional
        Any `scipy.integrate.solve_ivp` method. Default is "RK45".
    lyap_mode : {"none", "spectrum", "history"}, optional
        What to return besides the trajectory. Default is "spectrum".
    n_transient_steps : int, optional
        Transient length, in units of `dt_qr`. Default is 200.
    rtol, atol : float, optional
        Solver tolerances. Defaults are 1e-7 and 1e-9.
    **params
        Extra keyword arguments passed to `f` and `df` (e.g. A, K, sigma).

    Returns
    -------
    t_vals : np.ndarray
        Times, shape (n_intervals + 1,).
    traj : np.ndarray
        Trajectory sampled every `dt_qr`, shape (n_intervals + 1, d).
    spectrum : np.ndarray
        Only if lyap_mode="spectrum": Lyapunov exponents, largest first.
    lyap_times, lyap_history : np.ndarray
        Only if lyap_mode="history": running Lyapunov estimates over time.
    """
    # 1. Ensure 1D state vector
    y0 = np.asarray(y0, dtype=float).ravel()
    d = len(y0)

    # 2. Local closures that forward **params without burdening solve_ivp
    def f_eval(t, y):
        return np.asarray(f(t, y, **params), dtype=float).ravel()

    def df_eval(t, y):
        return np.asarray(df(t, y, **params), dtype=float)

    # Augmented RHS for simultaneously advancing the state and orthonormal frame Q
    def augmented_rhs(t, y_aug):
        y = y_aug[:d]
        Q_mat = y_aug[d:].reshape(d, d)

        dydt = f_eval(t, y)
        J = df_eval(t, y)
        dQdt = J @ Q_mat

        return np.concatenate([dydt, dQdt.ravel()])

    # 3. Transient burn-in (state only)
    t_transient = n_transient_steps * dt_qr
    current_t = 0.0

    if t_transient > 0:
        sol_trans = solve_ivp(
            f_eval,                     # Uses clean closure (t, y) -> no extra **params in solve_ivp
            (0.0, t_transient),
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        current_y = sol_trans.y[:, -1]
        current_t = sol_trans.t[-1]
    else:
        current_y = y0.copy()

    # 4. Main QR loop setup
    n_intervals = max(1, int(t_total / dt_qr))
    Q = np.eye(d, dtype=float)
    lyap_sum = np.zeros(d, dtype=float)

    traj = np.zeros((n_intervals + 1, d), dtype=float)
    t_vals = np.zeros(n_intervals + 1, dtype=float)
    traj[0] = current_y
    t_vals[0] = 0.0

    if lyap_mode == "history":
        lyap_history = np.zeros((n_intervals, d), dtype=float)
        lyap_times = np.zeros(n_intervals, dtype=float)

    elapsed_time = 0.0

    # 5. Iterative QR windows
    for step in range(n_intervals):
        y_aug_init = np.concatenate([current_y, Q.ravel()])
        t_span = (current_t, current_t + dt_qr)

        sol_step = solve_ivp(
            augmented_rhs,
            t_span,
            y_aug_init,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        current_y = sol_step.y[:d, -1]
        Q_evolved = sol_step.y[d:, -1].reshape(d, d)
        current_t = sol_step.t[-1]
        elapsed_time += dt_qr

        traj[step + 1] = current_y
        t_vals[step + 1] = elapsed_time

        # Re-orthonormalize
        Q, R = np.linalg.qr(Q_evolved)

        # Enforce positive diagonals to keep frame orientation consistent
        diag_r = np.diag(R)
        signs = np.sign(diag_r)
        signs[signs == 0] = 1.0
        Q *= signs

        abs_diag = np.maximum(np.abs(diag_r), 1e-300)
        lyap_sum += np.log(abs_diag)

        if lyap_mode == "history":
            lyap_history[step] = lyap_sum / elapsed_time
            lyap_times[step] = elapsed_time

    # 6. Final spectrum computation
    final_spectrum = np.sort(lyap_sum / elapsed_time)[::-1]

    if lyap_mode == "spectrum":
        return t_vals, traj, final_spectrum
    elif lyap_mode == "history":
        return t_vals, traj, lyap_times, lyap_history
    else:
        return t_vals, traj














    
    