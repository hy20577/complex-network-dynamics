"""Write the fixed input data shared by the Octave reference run and the tests.

Run from the project root:  python tests/reference/make_inputs.py
"""
from pathlib import Path

import numpy as np

from complex_network_dynamics import df_logistic, load_network, logistic, simulate_discrete_models

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


if __name__ == "__main__":
    main()
