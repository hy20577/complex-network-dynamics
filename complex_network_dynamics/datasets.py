from importlib.resources import files

import numpy as np

_DATA = files("complex_network_dynamics").joinpath("Data")


def load_network(name: str = "16nodes") -> np.ndarray:
    """Load one of the example networks that ship with the package.

    Two undirected, unweighted, connected networks are included:

    - "16nodes": 16 nodes, 26 links; the network of the 16-node examples in
      Yildirim & Antonopoulos (2024).
    - "12nodes": 12 nodes, 14 links; the network of the 12-node examples
      (file `12nodes.txt` of the thesis code).

    Parameters
    ----------
    name : str, optional
        Name of the network, with or without the file extension
        ("16nodes" or "16nodes.txt"). Default is "16nodes".

    Returns
    -------
    np.ndarray
        Adjacency matrix of floats, shape (M, M): 1 where two nodes are
        linked, 0 elsewhere; symmetric, with a zero diagonal.

    Raises
    ------
    FileNotFoundError
        If there is no network with this name. The message lists the
        available ones.
    """
    filename = name if name.endswith((".txt", ".csv")) else f"{name}.txt"
    file_ref = _DATA.joinpath(filename)

    if not file_ref.is_file():
        available = sorted(f.name.rsplit(".", 1)[0] for f in _DATA.iterdir() if f.name.endswith((".txt", ".csv")))
        raise FileNotFoundError(f"No network named '{name}' in the package. Available: {', '.join(available)}.")

    with file_ref.open("r", encoding="utf-8") as f:
        return np.loadtxt(f, delimiter="\t")
