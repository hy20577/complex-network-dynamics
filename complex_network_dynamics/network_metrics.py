import networkx as nx

def calculate_distance_metrics(G, mode="giant"):
    """Calculates diameter and average shortest path length for connected or

    unconnected graphs.

    Parameters
    ----------
    G : nx.Graph
        The input NetworkX graph.
    mode : str, optional (default='giant')
        - 'giant': Evaluates metrics strictly on the largest connected component.
        - 'all': Computes the maximum diameter across all components, and the
                 pair-weighted average path length over all reachable pairs.

    Returns
    -------
    diameter : int or float
        Diameter of the network/component. Returns 0 if no edges exist.
    avg_path_length : float
        Average shortest path length. Returns 0.0 if no edges exist.
    """
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
    elif mode == "all":
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

    else:
        raise ValueError("Mode must be either 'giant' or 'all'.")
