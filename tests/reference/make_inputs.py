"""Write the fixed input data shared by the Octave reference run and the tests.

Run from the project root:  python tests/reference/make_inputs.py
"""
from pathlib import Path

import numpy as np

from complex_network_dynamics import (
    amplitude_phase_surrogates,
    kuramoto,
    df_logistic,
    load_network,
    logistic,
    simulate_discrete_models,
)

DATA = Path(__file__).parent / "data"


def save(name, array):
    np.savetxt(DATA / name, np.atleast_2d(array), fmt="%.17g", delimiter=",")


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(20241002)

    # MIR family: 6 nodes of coupled logistic maps on the 16-node network
    A16 = load_network("16nodes")
    traj, _ = simulate_discrete_models(rng.random(16), logistic, df_logistic, A16, t=1000, alpha=0.1, r=4.0)
    save("in_mir_series.txt", traj[:, :6])

    # location: generic random points, and points that sit exactly on grid lines
    save("in_location_random.txt", rng.random((500, 2)))
    save("in_location_gridlines.txt", np.array([[0, 0], [0.25, 0.5], [0.5, 0.75], [0.75, 0.25], [1, 1]], dtype=float))

    # Coupled logistic map on the 12-node network
    save("in_logistic_adjacency.txt", load_network("12nodes"))
    save("in_logistic_x0.txt", rng.random(12))

    # Coupled logistic map with an isolated node (node 3 has no links)
    save("in_logistic_isolated_adjacency.txt", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float))
    save("in_logistic_isolated_x0.txt", np.array([0.2, 0.4, 0.7]))

    # Lorenz initial state
    save("in_lorenz_x0.txt", np.array([1.0, 1.0, 1.0]))

    # fdr_bh: plain, with ties and zeros, and with nothing significant
    save("in_fdr_pvals.txt", np.vstack([
        rng.random(20) ** 3,
        np.array([0, 0, 0.1, 0.1, 0.1, 0.2, 0.5, 0.5, 0.9, 1.0, 0, 0.3, 0.3, 0.05, 0.01, 0.01, 0.7, 0.8, 0.2, 0.4]),
        0.5 + 0.5 * rng.random(20),
    ]))

    # TPR_FPR: true network and an inferred network with some errors
    true_net = load_network("12nodes")
    inferred = true_net.copy()
    inferred[0, 1] = inferred[1, 0] = 0
    inferred[2, 7] = inferred[7, 2] = 1
    inferred[4, 9] = inferred[9, 4] = 1
    save("in_tpr_true.txt", true_net)
    save("in_tpr_inferred.txt", inferred)

    # Mir_surrogate_data: 5 nodes of the MIR series, their true links, and fixed
    # surrogates (10 per data length, lengths 500 and 1000) in call order
    series = np.loadtxt(DATA / "in_mir_series.txt", delimiter=",")[:, :5]
    save("in_surrogate_true_network.txt", A16[:5, :5])
    surrogates = amplitude_phase_surrogates(series, n_surrogates=20, rng=rng)
    save("in_surrogate_fixed.txt", surrogates.reshape(-1, 5))  # stacked: 20 blocks of 1000 rows

    # best_case_TPR_FPR: a full symmetric similarity matrix and the true network
    # (the upper-triangular MATLAB MIR output is also used, from ref_mir_auto_Mir.txt)
    save("in_best_case_true.txt", A16[:6, :6])
    sim = rng.random((6, 6))
    save("in_best_case_similarity.txt", sim + sim.T)

    # Synchronisation: phases of coupled Kuramoto oscillators (simple Euler steps)
    A6 = A16[:6, :6]
    w = rng.uniform(-1, 1, 6)
    theta = rng.random(6) * 2 * np.pi
    phases = np.empty((600, 6))
    for k in range(600):
        phases[k] = theta
        theta = theta + 0.05 * kuramoto(0.0, theta, A6, w, K=3.0)
    save("in_sync_phases.txt", phases)

    # Model vector fields and Jacobians: random states on the 12-node network
    save("in_model_state_kuramoto.txt", rng.random(12) * 2 * np.pi)
    save("in_model_w.txt", rng.uniform(0.5, 1.5, 12))
    save("in_model_state_3M.txt", rng.standard_normal(36) * 2)


if __name__ == "__main__":
    main()
