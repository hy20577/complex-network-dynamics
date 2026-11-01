import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numba import njit, prange
from matplotlib.ticker import MaxNLocator

from .surrogates import (
    amplitude_phase_surrogates,
    correlated_random_surrogates,
    phaseran,
    random_surrogates,
)



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
        

@njit(fastmath=True, cache=True)
def _cell_rows_cols(Dat, grid):
  """0-based row and column of each point of an (npts, 2) array on a grid x grid
  partition of its min-max normalised range. Shared by location_jit and Grid
  so both assign cells identically."""
  min_v1, max_v1 = Dat[:, 0].min(), Dat[:, 0].max()
  min_v2, max_v2 = Dat[:, 1].min(), Dat[:, 1].max()

  range_v1 = (max_v1 - min_v1) if max_v1 != min_v1 else 1.0
  range_v2 = (max_v2 - min_v2) if max_v2 != min_v2 else 1.0

  scaled_0 = ((Dat[:, 0] - min_v1) / range_v1) * grid
  scaled_1 = ((Dat[:, 1] - min_v2) / range_v2) * grid

  # Cells (a, b] like MATLAB's ceil(): a point exactly on a grid line belongs
  # to the lower cell, and the minimum (0) to the first cell
  rows = np.ceil(scaled_0).astype(np.int64) - 1
  cols = np.ceil(scaled_1).astype(np.int64) - 1

  # Robustly clamp integer indices to [0, grid - 1]
  rows = np.clip(rows, 0, grid - 1)
  cols = np.clip(cols, 0, grid - 1)
  return rows, cols


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
  rows, cols = _cell_rows_cols(Dat, grid)

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
  Dat = np.ascontiguousarray(Dat, dtype=np.float64)
  M = Dat.shape[1]
  pi, pj = np.triu_indices(M, k=1)
  N_final = _grid_sizes_per_pair(Dat, pi, pj)

  mean_N = np.mean(N_final)
  # Round halves up like MATLAB's round(); np.round would send 0.5 to 0
  N_min = int(np.floor(0.1 * mean_N + 0.5)) + 1
  N_max = int(np.floor(mean_N + 0.5))

  if N_max <= 2:
    N_max *= 2
  if N_min <= 2:
    N_min *= 2
  if N_max > 30:
    N_max = 30
  if N_min > N_max:
    N_min = N_max

  return N_min, N_max

@njit(parallel=True, cache=True)
def _grid_sizes_per_pair(Dat, pi, pj, inclusive=False):
  """For each pair, the largest grid size N whose average points per cell,
  npts / N^2, is not more than its number of occupied cells (eq. 17).
  With inclusive=True the search continues while they are equal, as in the
  Grid copy inside lagged_double_MIR.m."""
  npts = Dat.shape[0]
  N_final = np.empty(len(pi), dtype=np.int64)
  for k in prange(len(pi)):
    node12 = np.empty((npts, 2))
    node12[:, 0] = Dat[:, pi[k]]
    node12[:, 1] = Dat[:, pj[k]]

    N = 1
    N_c = 0
    mean_points = 1.0
    while mean_points > N_c or (inclusive and mean_points == N_c):
      N += 1
      rows, cols = _cell_rows_cols(node12, N)
      occupied = np.zeros(N * N, dtype=np.bool_)
      for i in range(npts):
        occupied[cols[i] + rows[i] * N] = True
      N_c = np.count_nonzero(occupied)
      mean_points = npts / (N**2)
    N_final[k] = N - 1
  return N_final


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
      Symmetric (M, M) matrix of decay times. The diagonal holds the decay
      time of each node paired with itself, which `MIR` uses as MATLAB does.
  """
  Dat = np.ascontiguousarray(Dat, dtype=np.float64)
  M = Dat.shape[1]
  pi, pj = np.triu_indices(M)
  _, decay = _pairs_mi_and_decay(Dat, pi, pj, int(gridsize))

  decay_matrix = np.zeros((M, M))
  decay_matrix[pi, pj] = decay
  decay_matrix[pj, pi] = decay
  return decay_matrix


# Neighbour lists of the transition graph in compressed form: the neighbours of
# cell u are neighbours[start[u]:start[u + 1]], in increasing order. A dense
# table is fastest for small grids (N <= 16), a sorted edge list for large ones.


@njit(cache=True)
def _neighbours_from_table(cells, nob):
  adjacent = np.zeros((nob, nob), dtype=np.bool_)
  for k in range(len(cells) - 1):
    u = cells[k]
    v = cells[k + 1]
    if u != v:
      adjacent[u, v] = True
      adjacent[v, u] = True
  start = np.zeros(nob + 1, dtype=np.int64)
  for u in range(nob):
    start[u + 1] = start[u] + np.count_nonzero(adjacent[u])
  neighbours = np.empty(start[nob], dtype=np.int64)
  k = 0
  for u in range(nob):
    for v in range(nob):
      if adjacent[u, v]:
        neighbours[k] = v
        k += 1
  return start, neighbours


@njit(cache=True)
def _neighbours_from_edge_list(cells, nob):
  # Each transition u -> v as two directed edges coded u * nob + v; the sorted
  # unique codes are the neighbour lists grouped by u
  codes = np.empty(2 * max(len(cells) - 1, 0), dtype=np.int64)
  m = 0
  for k in range(len(cells) - 1):
    u = cells[k]
    v = cells[k + 1]
    if u != v:
      codes[m] = u * nob + v
      codes[m + 1] = v * nob + u
      m += 2
  codes = np.unique(codes[:m])
  start = np.zeros(nob + 1, dtype=np.int64)
  for c in codes:
    start[c // nob + 1] += 1
  for u in range(nob):
    start[u + 1] += start[u]
  return start, codes % nob


@njit(cache=True)
def _decay_from_cells(cells, nob):
  """Largest shortest-path length in the graph of cell-to-cell transitions.

  Undirected and unweighted, self-transitions ignored; unreachable pairs do
  not count. Returns 1 when there are no transitions or all distances are 0.
  """
  if nob <= 256:
    start, neighbours = _neighbours_from_table(cells, nob)
  else:
    start, neighbours = _neighbours_from_edge_list(cells, nob)
  if start[nob] == 0:
    return 1.0

  # Breadth-first search from every cell that has a neighbour
  distance = np.full(nob, -1, dtype=np.int64)
  queue = np.empty(nob, dtype=np.int64)
  longest = 0
  for source in range(nob):
    if start[source + 1] == start[source]:
      continue
    distance[source] = 0
    queue[0] = source
    head = 0
    tail = 1
    while head < tail:
      u = queue[head]
      head += 1
      for k in range(start[u], start[u + 1]):
        v = neighbours[k]
        if distance[v] < 0:
          distance[v] = distance[u] + 1
          if distance[v] > longest:
            longest = distance[v]
          queue[tail] = v
          tail += 1
    for k in range(tail):  # reset only the cells this search reached
      distance[queue[k]] = -1

  return float(longest) if longest > 0 else 1.0


@njit(parallel=True, cache=True)
def _pairs_mi_and_decay(Dat, pi, pj, N):
  """MI and correlation decay time for the node pairs (pi[k], pj[k]) at grid size N.

  Each pair is binned once and the pairs run in parallel.
  """
  n_pairs = len(pi)
  npts = Dat.shape[0]
  mi = np.empty(n_pairs)
  decay = np.empty(n_pairs)
  for k in prange(n_pairs):
    node12 = np.empty((npts, 2))
    node12[:, 0] = Dat[:, pi[k]]
    node12[:, 1] = Dat[:, pj[k]]
    pair_mi, coordinates, _ = compute_single_pair_mi(node12, N)
    mi[k] = pair_mi
    decay[k] = _decay_from_cells(coordinates[:, 3].astype(np.int64), N * N)
  return mi, decay


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
  Dat = np.ascontiguousarray(Dat, dtype=np.float64)
  M = Dat.shape[1]
  pi, pj = np.triu_indices(M)  # pairs with i <= j, diagonal included

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

    mi, t = _pairs_mi_and_decay(Dat, pi, pj, int(N))
    I_s[pi, pj] = mi
    I_s[pj, pi] = mi
    I_c[pi, pj] = mi / t
    I_c[pj, pi] = mi / t

    if nodes is not None:
      Mi_MIR_over_partitions[0, idx] = I_s[nodes[0], nodes[1]]
      Mi_MIR_over_partitions[1, idx] = I_c[nodes[0], nodes[1]]

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


def fdr_bh(pvals, q=0.05, method="pdep"):
  """Benjamini-Hochberg false discovery rate procedure, MATLAB `fdr_bh`.

  Parameters
  ----------
  pvals : array_like
      p-values, any shape.
  q : float, optional
      False discovery rate. Default is 0.05.
  method : {"pdep", "dep"}, optional
      "pdep" assumes independent or positively dependent tests
      (Benjamini & Hochberg, 1995); "dep" allows any dependence
      (Benjamini & Yekutieli, 2001). Default is "pdep".

  Returns
  -------
  h : np.ndarray
      Boolean array, same shape as `pvals`: True where the null hypothesis
      is rejected.
  crit_p : float
      Largest p-value that is still significant (0 if none is).
  adj_ci_cvrg : float
      FDR-adjusted confidence interval coverage (NaN if nothing is significant).
  adj_p : np.ndarray
      FDR-adjusted p-values, same shape as `pvals`.
  """
  pvals = np.asarray(pvals, dtype=float)
  if np.any(pvals < 0):
    raise ValueError("Some p-values are less than 0.")
  if np.any(pvals > 1):
    raise ValueError("Some p-values are greater than 1.")

  flat = pvals.ravel()
  sort_ids = np.argsort(flat, kind="stable")
  p_sorted = flat[sort_ids]
  m = len(p_sorted)
  ranks = np.arange(1, m + 1)

  if method == "pdep":
    thresh = ranks * q / m
    wtd_p = m * p_sorted / ranks
  elif method == "dep":
    denom = m * np.sum(1.0 / ranks)
    thresh = ranks * q / denom
    wtd_p = denom * p_sorted / ranks
  else:
    raise ValueError("Argument 'method' needs to be 'pdep' or 'dep'.")

  # Adjusted p-values: each p gets the smallest weighted p at or after its rank
  adj_sorted = np.full(m, np.nan)
  wtd_sindex = np.argsort(wtd_p, kind="stable")
  nextfill = 0
  for k in range(m):
    if wtd_sindex[k] >= nextfill:
      adj_sorted[nextfill:wtd_sindex[k] + 1] = wtd_p[wtd_sindex[k]]
      nextfill = wtd_sindex[k] + 1
      if nextfill >= m:
        break
  adj_p = np.empty(m)
  adj_p[sort_ids] = adj_sorted

  rejected = np.nonzero(p_sorted <= thresh)[0]
  if len(rejected) == 0:
    return np.zeros(pvals.shape, dtype=bool), 0.0, np.nan, adj_p.reshape(pvals.shape)
  max_id = rejected[-1]
  crit_p = p_sorted[max_id]
  return pvals <= crit_p, crit_p, 1 - thresh[max_id], adj_p.reshape(pvals.shape)


def tpr_fpr(true_adjacency, inferred_adjacency):
  """True and false positive rates of an inferred network, MATLAB `TPR_FPR`.

  As in the MATLAB code, every entry of the matrices is counted, including
  the diagonal (where both are 0, so the diagonal adds true negatives).

  Parameters
  ----------
  true_adjacency : array_like
      Ground-truth binary adjacency matrix, shape (M, M).
  inferred_adjacency : array_like
      Inferred binary adjacency matrix, shape (M, M).

  Returns
  -------
  tpr : float
      TP / (TP + FN), or NaN if the true network has no links.
  fpr : float
      FP / (FP + TN), or NaN if the true network has no non-links.
  """
  A = np.asarray(true_adjacency) == 1
  B = np.asarray(inferred_adjacency) == 1
  A0 = np.asarray(true_adjacency) == 0
  B0 = np.asarray(inferred_adjacency) == 0
  tp = np.sum(B & A)
  fp = np.sum(B & A0)
  tn = np.sum(B0 & A0)
  fn = np.sum(B0 & A)
  tpr = tp / (tp + fn) if tp + fn > 0 else np.nan
  fpr = fp / (fp + tn) if fp + tn > 0 else np.nan
  return tpr, fpr


_SURROGATE_GENERATORS = {
    "amplitude_phase": lambda data, rng, pc_a, pc_p: amplitude_phase_surrogates(data, 1, pc_a, pc_p, rng)[0],
    "phaseran": lambda data, rng, pc_a, pc_p: phaseran(data, 1, rng)[0],
    "random": lambda data, rng, pc_a, pc_p: random_surrogates(data, 1, rng)[0],
    "correlated_random": lambda data, rng, pc_a, pc_p: correlated_random_surrogates(data, 1, rng=rng)[0],
}
# Names used by the MATLAB code
_SURROGATE_GENERATORS["HT_mix_A_theta"] = _SURROGATE_GENERATORS["amplitude_phase"]
_SURROGATE_GENERATORS["twnsd"] = _SURROGATE_GENERATORS["phaseran"]
_SURROGATE_GENERATORS["randsd"] = _SURROGATE_GENERATORS["random"]
_SURROGATE_GENERATORS["corr_randsd"] = _SURROGATE_GENERATORS["correlated_random"]


def mir_surrogate_test(data, p_sl=0.01, time_range=None, surrogate="amplitude_phase", true_network=None,
                       pc_amplitude=1.0, pc_phase=0.5, fdr_q=None, rng=None):
  """Infer a network with MIR and a surrogate data test, MATLAB `Mir_surrogate_data`.

  For each data length in `time_range`, MIR is computed for every pair of
  nodes in the data and in 1/p_sl surrogate datasets. A pair's p-value is the
  fraction of surrogates whose MIR is at least the MIR of the data. The
  Benjamini-Hochberg procedure (`fdr_bh`) then decides which pairs are linked.

  Parameters
  ----------
  data : array_like
      Time series after the transient, shape (n_points, n_nodes).
  p_sl : float, optional
      Significance level; also sets the number of surrogates, 1/p_sl.
      Default is 0.01 (100 surrogates).
  time_range : sequence of int, optional
      Data lengths to test, each using the first that many points.
      Default is the full length only.
  surrogate : str or callable, optional
      "amplitude_phase" (APMSD, MATLAB "HT_mix_A_theta"), "phaseran"
      ("twnsd"), "random" ("randsd") or "correlated_random" ("corr_randsd"),
      or a function f(data, rng) returning one surrogate of shape
      (n_points, n_nodes). Default is "amplitude_phase".
  true_network : array_like, optional
      Ground-truth adjacency matrix. If given, TPR and FPR are computed.
  pc_amplitude, pc_phase : float, optional
      Fractions of time points whose amplitudes and phases are randomised
      by the "amplitude_phase" surrogates. Defaults are 1 and 0.5.
  fdr_q : float, optional
      False discovery rate for `fdr_bh`. Default is p_sl / 100, as in the
      MATLAB code.
  rng : numpy.random.Generator or int, optional
      Random generator or seed for the surrogates.

  Returns
  -------
  mir_overall : np.ndarray
      Shape (len(time_range), 1 + n_surrogates, n_pairs). Row 0 holds the MIR
      of the data, the other rows the MIR of each surrogate. Pairs are in the
      order (0, 1), (0, 2), ..., (1, 2), ...
  adjacency : np.ndarray
      Inferred symmetric adjacency matrices after FDR, shape
      (len(time_range), n_nodes, n_nodes).
  tp_fp_rate : np.ndarray
      Shape (len(time_range), 3): data length, TPR and FPR (NaN without
      `true_network`).
  """
  data = np.asarray(data, dtype=float)
  n_points, n_nodes = data.shape
  rng = np.random.default_rng(rng)
  n_surrogates = int(round(1 / p_sl))
  fdr_q = p_sl / 100 if fdr_q is None else fdr_q
  time_range = [n_points] if time_range is None else [min(int(t), n_points) for t in time_range]

  if callable(surrogate):
    make_surrogate = lambda d: surrogate(d, rng)
  elif surrogate in _SURROGATE_GENERATORS:
    generator = _SURROGATE_GENERATORS[surrogate]
    make_surrogate = lambda d: generator(d, rng, pc_amplitude, pc_phase)
  else:
    raise ValueError(f"Unknown surrogate type '{surrogate}'.")

  pairs = np.triu_indices(n_nodes, k=1)
  n_pairs = len(pairs[0])
  mir_overall = np.zeros((len(time_range), 1 + n_surrogates, n_pairs))
  adjacency = np.zeros((len(time_range), n_nodes, n_nodes))
  tp_fp_rate = np.full((len(time_range), 3), np.nan)
  tp_fp_rate[:, 0] = time_range

  for it, length in enumerate(time_range):
    segment = data[:length]
    mir_overall[it, 0] = MIR(segment)[0][pairs]
    for k in range(n_surrogates):
      mir_overall[it, 1 + k] = MIR(make_surrogate(segment))[0][pairs]

    p_values = np.sum(mir_overall[it, 1:] >= mir_overall[it, 0], axis=0) / n_surrogates
    h, _, _, _ = fdr_bh(p_values, fdr_q, "pdep")
    adjacency[it][pairs] = h
    adjacency[it] += adjacency[it].T

    if true_network is not None:
      tp_fp_rate[it, 1:] = tpr_fpr(true_network, adjacency[it])

  return mir_overall, adjacency, tp_fp_rate


def _rescale_upper(I_c, guard_equal):
  """Min-max scale the upper triangle of I_c, as in the MATLAB double-normalised
  MIR: the minimum is taken over the nonzero entries of the whole matrix."""
  M = len(I_c)
  hat = np.zeros((M, M))
  nonzero = I_c[I_c != 0]
  if nonzero.size == 0:
    return hat
  hi, lo = I_c.max(), nonzero.min()
  if guard_equal and hi == lo:
    return hat
  upper = np.triu_indices(M, k=1)
  with np.errstate(divide="ignore", invalid="ignore"):
    hat[upper] = (I_c[upper] - lo) / (hi - lo)
  return hat


def double_normalised_mir(Dat, true_network=None):
  """Double-normalised MIR, MATLAB `double_normalised_MIR`.

  The data are scaled to [0, 1]. For each grid size from `Grid`, the MIR of
  all pairs is min-max scaled; the scaled matrices are summed over grid sizes
  and divided by their largest entry.

  Parameters
  ----------
  Dat : array_like
      Time series, shape (npts, M).
  true_network : array_like, optional
      Ground-truth adjacency matrix. If given, the best reconstruction rate
      over thresholds 0.01, 0.02, ..., 1 is returned.

  Returns
  -------
  normalised_MIR : np.ndarray
      Symmetric (M, M) matrix with values up to 1; diagonal 0.
  max_rate : float or None
      Largest reconstruction percentage, 100 - 100 * (mismatched entries) /
      (number of links counted in both directions), floored at 0.
      None if `true_network` is not given.
  """
  Dat = np.asarray(Dat, dtype=np.float64)
  Dat = (Dat - Dat.min(axis=0)) / (Dat.max(axis=0) - Dat.min(axis=0))
  Dat = np.ascontiguousarray(Dat)
  M = Dat.shape[1]
  pi, pj = np.triu_indices(M, k=1)
  N_min, N_max = Grid(Dat)

  total = np.zeros((M, M))
  for N in range(N_min, N_max + 1):
    mi, t = _pairs_mi_and_decay(Dat, pi, pj, N)
    I_c = np.zeros((M, M))
    I_c[pi, pj] = mi / t
    total += _rescale_upper(I_c, guard_equal=True)

  normalised = total / total.max()
  normalised = normalised + normalised.T

  if true_network is None:
    return normalised, None
  A = np.asarray(true_network, dtype=float)
  rates = np.empty(100)
  for k in range(1, 101):
    adj = normalised.copy()
    adj[pi, pj] = adj[pj, pi] = normalised[pi, pj] >= k / 100
    rates[k - 1] = 100 - np.sum(np.abs(adj - A) == 1) * (100 / A.sum())
  return normalised, max(rates.max(), 0.0)


@njit(parallel=True, cache=True)
def _pairs_lagged_mi_and_decay(Dat, pi, pj, N, max_lag):
  """Largest MI over lags -max_lag..max_lag and the zero-lag decay time, per pair."""
  n = Dat.shape[0]
  best = np.empty(len(pi))
  decay = np.empty(len(pi))
  for k in prange(len(pi)):
    x = Dat[:, pi[k]]
    y = Dat[:, pj[k]]
    node12 = np.empty((n, 2))
    node12[:, 0] = x
    node12[:, 1] = y
    largest, coordinates, _ = compute_single_pair_mi(node12, N)
    decay[k] = _decay_from_cells(coordinates[:, 3].astype(np.int64), N * N)
    for tau in range(1, max_lag + 1):
      shifted = np.empty((n - tau, 2))
      # Negative lag: node i leads
      shifted[:, 0] = x[: n - tau]
      shifted[:, 1] = y[tau:]
      value, _, _ = compute_single_pair_mi(shifted, N)
      largest = max(largest, value)
      # Positive lag: node j leads
      shifted[:, 0] = x[tau:]
      shifted[:, 1] = y[: n - tau]
      value, _, _ = compute_single_pair_mi(shifted, N)
      largest = max(largest, value)
    best[k] = largest
  return best, decay


def lagged_double_mir(Dat, max_lag):
  """Lagged, double-normalised MIR, MATLAB `lagged_double_MIR`.

  For each pair, the MI is the largest MI over time lags -max_lag..max_lag,
  divided by the zero-lag correlation decay time, then min-max scaled and
  summed over grid sizes as in `double_normalised_mir`.

  The grid sizes follow the copy of Grid inside `lagged_double_MIR.m`, which
  differs from `Grid`: its search continues while the average points per cell
  equals the number of occupied cells, and N_max is not capped at 30.

  Parameters
  ----------
  Dat : array_like
      Time series, shape (npts, M).
  max_lag : int
      Largest lag, in time steps.

  Returns
  -------
  lagged_MI : np.ndarray
      Symmetric (M, M) matrix of the largest lagged MI, summed (not averaged)
      over grid sizes, as in the MATLAB code.
  lagged_MIR : np.ndarray
      Symmetric (M, M) double-normalised MIR, values up to 1.
  """
  Dat = np.ascontiguousarray(Dat, dtype=np.float64)
  M = Dat.shape[1]
  pi, pj = np.triu_indices(M, k=1)

  N_final = _grid_sizes_per_pair(Dat, pi, pj, True)
  mean_N = np.mean(N_final)
  N_min = int(np.floor(0.1 * mean_N + 0.5)) + 1
  N_max = int(np.floor(mean_N + 0.5))
  if N_max <= 2:
    N_max *= 2
  if N_min <= 2:
    N_min *= 2

  total_mir = np.zeros((M, M))
  total_mi = np.zeros((M, M))
  for N in range(N_min, N_max + 1):
    mi, t = _pairs_lagged_mi_and_decay(Dat, pi, pj, N, int(max_lag))
    I_s = np.zeros((M, M))
    I_c = np.zeros((M, M))
    I_s[pi, pj] = mi
    I_c[pi, pj] = mi / t
    total_mir += _rescale_upper(I_c, guard_equal=False)
    total_mi += I_s

  lagged_MIR = total_mir / total_mir.max()
  return total_mi + total_mi.T, lagged_MIR + lagged_MIR.T


def best_case_tpr_fpr(similarity_matrix, true_adjacency):
  """ROC sweep over thresholds and the best threshold, MATLAB `best_case_TPR_FPR`.

  Every pairwise similarity (minus 1e-12) is tried as a threshold; entries
  above it count as links. The best threshold is the one closest to the
  perfect point (FPR, TPR) = (0, 1). As in the MATLAB code the whole matrix is
  counted, diagonal included, and an upper-triangular input is made symmetric.

  Parameters
  ----------
  similarity_matrix : array_like
      (M, M) similarity matrix, e.g. MIR. Its diagonal is ignored.
  true_adjacency : array_like
      (M, M) ground-truth binary adjacency matrix.

  Returns
  -------
  roc : np.ndarray
      Shape (M(M-1)/2, 4), one row per threshold from high to low: TPR, FPR,
      distance to (0, 1), threshold.
  best : np.ndarray
      The rows of `roc` with the smallest distance.
  inferred : np.ndarray
      Inferred adjacency matrix at the best threshold (the smallest one if
      several are equally good): pairs with similarity at least the threshold.
  """
  S = np.array(similarity_matrix, dtype=float)
  A = np.asarray(true_adjacency)
  M = len(S)
  np.fill_diagonal(S, 0)
  if np.array_equal(S, np.triu(S)):
    S = S + S.T
  upper = np.triu_indices(M, k=1)
  thresholds = np.sort(S[upper])[::-1] - 1e-12

  roc = np.empty((len(thresholds), 4))
  for k, th in enumerate(thresholds):
    tpr, fpr = tpr_fpr(A, S > th)
    roc[k] = tpr, fpr, np.sqrt((tpr - 1) ** 2 + fpr ** 2), th
  best = roc[roc[:, 2] == roc[:, 2].min()]

  th = best[:, 3].min()
  inferred = S.copy()
  inferred[upper] = S[upper] >= th
  inferred[(upper[1], upper[0])] = inferred[upper]
  return roc, best, inferred
