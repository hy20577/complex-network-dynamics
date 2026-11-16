"""Tests for the bundled networks (load_network) and calculate_distance_metrics."""
import networkx as nx
import numpy as np
import pytest

from complex_network_dynamics import calculate_distance_metrics, load_network


@pytest.mark.parametrize("name, n_nodes, n_links", [("12nodes", 12, 14), ("16nodes", 16, 26)])
def test_bundled_networks(name, n_nodes, n_links):
    A = load_network(name)
    assert A.shape == (n_nodes, n_nodes)
    assert set(np.unique(A)) == {0.0, 1.0}
    np.testing.assert_array_equal(A, A.T)
    assert not A.diagonal().any()
    assert A.sum() == 2 * n_links
    assert nx.is_connected(nx.from_numpy_array(A))


def test_load_network_default_and_extension():
    np.testing.assert_array_equal(load_network(), load_network("16nodes"))
    np.testing.assert_array_equal(load_network("16nodes.txt"), load_network("16nodes"))


def test_load_network_unknown_name_lists_the_available_ones():
    with pytest.raises(FileNotFoundError, match="12nodes, 16nodes"):
        load_network("99nodes")


def test_distance_metrics_connected_matches_networkx():
    G = nx.from_numpy_array(load_network("12nodes"))
    for mode in ("giant", "all"):
        diameter, avg_path = calculate_distance_metrics(G, mode)
        assert diameter == nx.diameter(G)
        assert avg_path == pytest.approx(nx.average_shortest_path_length(G))


def test_distance_metrics_disconnected():
    # A path of 4 nodes (distances 1, 1, 1, 2, 2, 3), a triangle (1, 1, 1) and an isolated node
    G = nx.disjoint_union(nx.disjoint_union(nx.path_graph(4), nx.complete_graph(3)), nx.empty_graph(1))
    assert calculate_distance_metrics(G) == (3, pytest.approx(10 / 6))
    assert calculate_distance_metrics(G, "all") == (3, pytest.approx(13 / 9))


@pytest.mark.parametrize("G", [nx.empty_graph(0), nx.empty_graph(1), nx.empty_graph(5)])
def test_distance_metrics_without_links(G):
    assert calculate_distance_metrics(G) == (0, 0.0)
    assert calculate_distance_metrics(G, "all") == (0, 0.0)


def test_distance_metrics_rejects_bad_input():
    with pytest.raises(ValueError, match="mode"):
        calculate_distance_metrics(nx.path_graph(3), "largest")  # also when the network is connected
    with pytest.raises(ValueError, match="undirected"):
        calculate_distance_metrics(nx.DiGraph([(0, 1)]))
