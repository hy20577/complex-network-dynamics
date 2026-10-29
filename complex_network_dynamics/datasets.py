# complex_network_dynamics/datasets.py
from importlib.resources import files
import numpy as np

def load_network(name: str = "16nodes") -> np.ndarray:
    """
    Load a built-in network matrix from the package Data directory.
    
    Parameters
    ----------
    name : str
        Name of the dataset file (with or without extension, e.g., '16nodes' or '16nodes.txt').
        
    Returns
    -------
    np.ndarray
        Adjacency matrix.
    """
    # Auto-append extension if omitted
    filename = name if name.endswith((".txt", ".csv")) else f"{name}.txt"
    
    file_ref = files("complex_network_dynamics").joinpath("Data", filename)
    
    if not file_ref.is_file():
        raise FileNotFoundError(f"Dataset '{filename}' not found in package Data folder.")
        
    with file_ref.open("r", encoding="utf-8") as f:
        return np.loadtxt(f, delimiter="\t")