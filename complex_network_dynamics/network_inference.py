import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from numba import njit
from matplotlib.ticker import MaxNLocator



def plot_similarity_gap_bars(similarity_matrix, true_adjacency, sort_bars=True, title=None):
    """
    Plots a bar chart of similarity values, coloring true vs false edges,
    and highlights the exact threshold band for 100% perfect network inference.
    
    Parameters
    ----------
    similarity_matrix : np.ndarray
        (N, N) similarity matrix.
    true_adjacency : np.ndarray
        (N, N) ground-truth binary adjacency matrix.
    sort_bars : bool, optional
        If True, sorts the bars in descending order of similarity. 
        If False, plots them in the original flattened pair index order. Default is True.
    title : str, optional
        Title for the plot. If None, defaults to 'Pairwise Similarities'. Default is None.

    Returns
    -------
    tuple of float or None
        (max_false_sim, min_true_sim): the band of thresholds that separates
        true edges from non-edges perfectly, or None if no such band exists.
    """
    N = similarity_matrix.shape[0]

    # 1. Extract upper triangle (exclude diagonal self-connections)
    triu_indices = np.triu_indices(N, k=1)
    sim_values = similarity_matrix[triu_indices]
    true_edges = (true_adjacency[triu_indices] != 0).astype(int)

    # 2. Apply sorting condition
    if sort_bars:
        sort_idx = np.argsort(sim_values)[::-1]
        plot_sims = sim_values[sort_idx]
        plot_truth = true_edges[sort_idx]
        x_label = "Node Pairs (Ranked highest to lowest similarity)"
    else:
        plot_sims = sim_values
        plot_truth = true_edges
        x_label = "Node Pairs (Unsorted upper-triangle order)"

    # 3. Find the critical boundaries for 100% accuracy
    sim_true = sim_values[true_edges == 1]
    sim_false = sim_values[true_edges == 0]
    
    min_true_sim = np.min(sim_true) if len(sim_true) > 0 else 0.0
    max_false_sim = np.max(sim_false) if len(sim_false) > 0 else 0.0
    
    is_perfect_possible = min_true_sim > max_false_sim

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(plot_sims))
    
    # Map colors: Green for True Edges, Red for False Edges (Noise/Spurious)
    colors = ['#2ca02c' if truth == 1 else '#d62728' for truth in plot_truth]
    
    plt.bar(x_pos, plot_sims, color=colors, edgecolor='black', linewidth=0.5, width=1.0)

    # 5. Draw the Grey Threshold Band (or indicate overlap)
    if is_perfect_possible:
        # The grey band where any threshold yields 100% correct inference
        plt.axhspan(max_false_sim, min_true_sim, color='grey', alpha=0.35, 
                    label=f'100% Correct Band\n({max_false_sim:.3f}, {min_true_sim:.3f}]')
        
        plt.axhline(min_true_sim, color='#2ca02c', linestyle='--', alpha=0.8)
        plt.axhline(max_false_sim, color='#d62728', linestyle='--', alpha=0.8)
    else:
        # If max false is higher than min true, they overlap. Perfect inference is mathematically impossible.
        plt.axhline(min_true_sim, color='#2ca02c', linestyle='--', alpha=0.8, label=f'Min True Edge ({min_true_sim:.3f})')
        plt.axhline(max_false_sim, color='#d62728', linestyle='--', alpha=0.8, label=f'Max False Edge ({max_false_sim:.3f})')

    # Formatting
    plot_title = title if title is not None else "Pairwise Similarities"
    plt.title(plot_title, fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Similarity Value", fontsize=11)
    plt.xlabel(x_label, fontsize=11)
    
    # Custom Legend
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Ground-Truth Edge'),
        Patch(facecolor='#d62728', edgecolor='black', label='Spurious Connection')
    ]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=legend_elements + handles, loc='upper right', frameon=True)

    # Remove x-ticks since it's just an index of pairs
    plt.xticks([]) 
    plt.xlim(-0.5, len(plot_sims) - 0.5)
    plt.tight_layout()
    plt.show()

    if is_perfect_possible:
        return (max_false_sim, min_true_sim)
    else:
        print("Warning: Topologies overlap. No threshold can perfectly reconstruct the network.")
        return None
        

@njit(fastmath=True)
def location_jit(Dat, grid):
  """JIT-compiled spatial grid binning.

  Normalizes a 2D point cloud to the unit square and assigns each point to a
  cell of a `grid` x `grid` partition.

  Parameters
  ----------
  Dat : np.ndarray
      Points, shape (npts, 2).
  grid : int
      Number of cells per axis.

  Returns
  -------
  coordinates : np.ndarray
      Shape (npts, 4): point index, row (1-based), column (1-based) and flat
      cell index (0-based).
  ppb : np.ndarray
      Points per box, shape (grid**2, 2): cell index and point count.
  """
  npts = Dat.shape[0]

  # Min-Max Normalization
  min_v1, max_v1 = Dat[:, 0].min(), Dat[:, 0].max()
  min_v2, max_v2 = Dat[:, 1].min(), Dat[:, 1].max()

  range_v1 = (max_v1 - min_v1) if max_v1 != min_v1 else 1.0
  range_v2 = (max_v2 - min_v2) if max_v2 != min_v2 else 1.0

  scaled_0 = ((Dat[:, 0] - min_v1) / range_v1) * grid
  scaled_1 = ((Dat[:, 1] - min_v2) / range_v2) * grid

  rows = np.floor(scaled_0).astype(np.int64)
  cols = np.floor(scaled_1).astype(np.int64)

  # Robustly clamp integer indices to [0, grid - 1]
  rows = np.clip(rows, 0, grid - 1)
  cols = np.clip(cols, 0, grid - 1)

  # Flat cell index: guaranteed 0 to (grid^2 - 1)
  flat_indices = cols + rows * grid

  coordinates = np.empty((npts, 4), dtype=np.float64)
  for i in range(npts):
    coordinates[i, 0] = i
    coordinates[i, 1] = rows[i] + 1  # 1-based [1, grid]
    coordinates[i, 2] = cols[i] + 1  # 1-based [1, grid]
    coordinates[i, 3] = flat_indices[i]  # 0-based flat index [0, grid^2 - 1]

  nob = grid * grid
  boxcounter = np.zeros(nob, dtype=np.float64)
  for i in range(npts):
    boxcounter[flat_indices[i]] += 1.0

  ppb = np.empty((nob, 2), dtype=np.float64)
  for b in range(nob):
    ppb[b, 0] = b
    ppb[b, 1] = boxcounter[b]

  return coordinates, ppb


def Grid(Dat):
  """Computes optimal partition bounds [N_min, N_max] based on

  average occupancy criteria across node pairs.

  Parameters
  ----------
  Dat : array_like
      Time series, shape (npts, M): one column per node.

  Returns
  -------
  N_min, N_max : int
      Smallest and largest grid size to use in `MIR` (N_max is capped at 30).
  """
  Dat = np.asarray(Dat)
  npts, M = Dat.shape

  num_pairs = M * (M - 1) // 2
  N_final = np.zeros(num_pairs, dtype=int)

  r = 0
  for k in range(M - 1):
    for m in range(k + 1, M):
      node12 = np.column_stack([Dat[:, k], Dat[:, m]])

      N = 1
      N_c = 0
      mean_points = 1.0

      while mean_points > N_c:
        N += 1
        _, ppb = location_jit(node12, N)
        N_c = np.count_nonzero(ppb[:, 1])
        mean_points = npts / (N**2)

      N_final[r] = N - 1
      r += 1

  mean_N = np.mean(N_final)
  N_min = int(np.round(0.1 * mean_N)) + 1
  N_max = int(np.round(mean_N))

  if N_max <= 2:
    N_max *= 2
  if N_min <= 2:
    N_min *= 2
  if N_max > 30:
    N_max = 30
  if N_min > N_max:
    N_min = N_max

  return N_min, N_max

def corr_decay(Dat, gridsize):
  """Correlation decay time for every pair of nodes.

  For each pair, the joint trajectory is binned on a `gridsize` x `gridsize`
  grid and the cell-to-cell transitions form a graph. The largest shortest
  path in that graph estimates how long correlations take to decay.

  Parameters
  ----------
  Dat : array_like
      Time series, shape (npts, M).
  gridsize : int
      Number of cells per axis.

  Returns
  -------
  np.ndarray
      Symmetric (M, M) matrix of decay times (diagonal is 0).
  """
  Dat = np.asarray(Dat)
  M = Dat.shape[1]
  nob = gridsize**2

  decay_matrix = np.zeros((M, M))

  for i in range(M):
    for j in range(i + 1, M):
      node12 = np.column_stack([Dat[:, i], Dat[:, j]])
      coordinates, _ = location_jit(node12, gridsize)

      cell_seq = coordinates[:, 3].astype(np.int64)
      u = cell_seq[:-1]
      v = cell_seq[1:]

      # Remove self-transitions
      mask = u != v
      u_filtered = u[mask]
      v_filtered = v[mask]

      if len(u_filtered) == 0:
        decay_matrix[i, j] = 1.0
        decay_matrix[j, i] = 1.0
        continue

      data = np.ones(len(u_filtered), dtype=bool)
      adj_sparse = csr_matrix(
          (data, (u_filtered, v_filtered)), shape=(nob, nob)
      )
      adj_sparse = adj_sparse + adj_sparse.T

      dist_matrix = shortest_path(
          csgraph=adj_sparse, directed=False, unweighted=True
      )
      dist_matrix[np.isinf(dist_matrix)] = 0.0

      max_dist = dist_matrix.max()
      d_val = max_dist if max_dist > 0 else 1.0
      decay_matrix[i, j] = d_val
      decay_matrix[j, i] = d_val  # Symmetrize

  return decay_matrix


@njit(fastmath=True)
def compute_single_pair_mi(node12, N):
  """JIT-compiled computation of Shannon Mutual Information for a single pair at

  partition N.

  Precomputes 1D marginal frequencies in O(npts) instead of O(npts * N^2).

  Parameters
  ----------
  node12 : np.ndarray
      Joint time series of two nodes, shape (npts, 2).
  N : int
      Number of cells per axis.

  Returns
  -------
  MI : float
      Mutual information (natural log).
  coordinates, ppb : np.ndarray
      Binning output from `location_jit`.
  """
  coordinates, ppb = location_jit(node12, N)
  n_pts = node12.shape[0]
  nob = N * N

  # 1. Precalculate 1D marginal counts in a single O(n_pts) pass
  count_r = np.zeros(N + 1, dtype=np.int64)
  count_c = np.zeros(N + 1, dtype=np.int64)
  for idx in range(n_pts):
    count_r[int(coordinates[idx, 1])] += 1
    count_c[int(coordinates[idx, 2])] += 1

  MI = 0.0
  inv_npts = 1.0 / n_pts

  # 2. Iterate over boxes in O(N^2)
  for m in range(1, nob + 1):
    joint_prob = ppb[m - 1, 1] * inv_npts
    if joint_prob > 0.0:
      target_row = int(np.ceil(m / N))
      target_col = N if (m % N == 0) else (m % N)

      prob_node1 = count_r[target_row] * inv_npts
      prob_node2 = count_c[target_col] * inv_npts

      if prob_node1 > 0.0 and prob_node2 > 0.0:
        MI += joint_prob * np.log(joint_prob / (prob_node1 * prob_node2))

  return MI, coordinates, ppb


def MIR(Dat, N_range=None, nodes=None):
  """Mutual Information Rate (MIR) between every pair of nodes.

  For each grid size N in the partition range, MIR is the mutual information
  divided by the correlation decay time (`corr_decay`). Results are averaged
  over all grid sizes. Higher MIR suggests a direct link.

  Parameters
  ----------
  Dat : array_like
      Time series, shape (npts, M): one column per node.
  N_range : tuple of int, optional
      (N_min, N_max) grid sizes. If None, chosen automatically by `Grid`.
  nodes : tuple of int, optional
      (i, j) with i <= j. If given, plots how MI and MIR for this pair
      change with the grid size.

  Returns
  -------
  Mir : np.ndarray
      MIR matrix, shape (M, M).
  Mi : np.ndarray
      Mutual information matrix, shape (M, M).
  """
  Dat = np.asarray(Dat, dtype=np.float64)
  M = Dat.shape[1]

  if N_range is None:
    N_min, N_max = Grid(Dat)
  else:
    N_min, N_max = int(N_range[0]), int(N_range[1])

  partition_range = np.arange(N_min, N_max + 1)
  num_partitions = len(partition_range)

  Is = np.zeros((M, M))
  Ic = np.zeros((M, M))
  Mi_MIR_over_partitions = np.zeros((2, num_partitions))

  for idx, N in enumerate(partition_range):
    I_s = np.zeros((M, M))
    I_c = np.zeros((M, M))

    decay_mat = corr_decay(Dat, N)

    for i in range(M):
      for j in range(i, M):
        node12 = np.column_stack([Dat[:, i], Dat[:, j]])
        MI, _, _ = compute_single_pair_mi(node12, N)

        t = decay_mat[i, j] if decay_mat[i, j] > 0 else 1.0

        I_s[i, j] = MI
        I_s[j, i] = MI
        I_c[i, j] = MI / t
        I_c[j, i] = MI / t

        if nodes is not None and nodes[0] == i and nodes[1] == j:
          Mi_MIR_over_partitions[0, idx] = MI
          Mi_MIR_over_partitions[1, idx] = MI / t

    Is += I_s
    Ic += I_c

  Mir = Ic / num_partitions
  Mi = Is / num_partitions

  if nodes is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MI plot (linear x, log y)
    axes[0].semilogy(partition_range, Mi_MIR_over_partitions[0, :], "o--")
    axes[0].set_title(f"Evolution of MI for nodes {nodes} over partitions")
    axes[0].set_xlabel("Partition Size N")
    axes[0].set_ylabel("MI")
    axes[0].grid(True, which="both", ls="--")

    # MIR plot (linear x, log y)
    axes[1].semilogy(partition_range, Mi_MIR_over_partitions[1, :], ".-")
    axes[1].set_title(f"Evolution of MIR for nodes {nodes} over partitions")
    axes[1].set_xlabel("Partition Size N")
    axes[1].set_ylabel("MIR")
    axes[1].grid(True, which="both", ls="--")

    # Enforce integer ticks on x-axis for both subplots
    for ax in axes:
      if len(partition_range) <= 15:
        ax.set_xticks(partition_range)
      else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

  return Mir, Mi         
        