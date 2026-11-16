import networkx as nx

def calculate_distance_metrics(G, mode="giant"):
    """Diameter and average shortest path length of a network that may be disconnected.

    NetworkX's `diameter` and `average_shortest_path_length` raise an error
    when the network is not connected, because some distances are infinite.
    This function restricts both to the pairs of nodes that can reach each
    other. For a connected network it returns the NetworkX values.

    Parameters
    ----------
    G : networkx.Graph
        Undirected network.
    mode : {"giant", "all"}, optional
        What to do when the network is not connected. "giant" (default) uses
        the largest connected component only. "all" returns the largest
        diameter over all components, and the average shortest path length
        over every pair of nodes that can reach each other.

    Returns
    -------
    diameter : int
        Longest shortest path. 0 if the network has no links.
    avg_path_length : float
        Average shortest path length. 0.0 if the network has no links.

    Raises
    ------
    ValueError
        If `mode` is not "giant" or "all", or `G` is directed.
    """
    if mode not in ("giant", "all"):
        raise ValueError(f"mode must be 'giant' or 'all'; got '{mode}'.")
    if G.is_directed():
        raise ValueError("G must be undirected; for a directed network, pass G.to_undirected().")
    if len(G) == 0:
        return 0, 0.0

    # Case 1: Fully connected graph
    if nx.is_connected(G):
        if len(G) <= 1:
            return 0, 0.0
        return nx.diameter(G), nx.average_shortest_path_length(G)

    # Filter connected components that have at least 2 nodes
    components = [
        G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1
    ]

    # Handle edge case: no edges exist in the graph (all isolated nodes)
    if not components:
        return 0, 0.0

    # Case 2: Unconnected - Giant Component Mode (Standard)
    if mode == "giant":
        giant = max(components, key=len)
        return nx.diameter(giant), nx.average_shortest_path_length(giant)

    # Case 3: Unconnected - Aggregate Across All Components Mode
    else:
        # Global diameter is the maximum diameter found in any component
        diam = max(nx.diameter(c) for c in components)

        # Weighted average shortest path length over all connected pairs
        total_reachable_pairs = 0
        weighted_path_sum = 0.0

        for c in components:
            n_c = len(c)
            pairs_c = n_c * (n_c - 1) / 2
            l_c = nx.average_shortest_path_length(c)

            weighted_path_sum += l_c * pairs_c
            total_reachable_pairs += pairs_c

        avg_path = (
            (weighted_path_sum / total_reachable_pairs)
            if total_reachable_pairs > 0
            else 0.0
        )
        return diam, avg_path
