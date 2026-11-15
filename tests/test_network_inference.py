"""Input checks and conventions of the network-inference functions."""
import numpy as np
import pytest

from complex_network_dynamics import fdr_bh, load_network, mir_surrogate_test, tpr_fpr


def test_tpr_fpr_counts_the_whole_matrix_as_the_thesis_code():
    A = load_network("16nodes")
    inferred = A.copy()
    i, j = next((i, j) for i in range(16) for j in range(i + 1, 16) if A[i, j] == 0)
    inferred[i, j] = inferred[j, i] = 1
    n_nonlinked_pairs = 120 - int(A.sum()) // 2
    tpr, fpr = tpr_fpr(A, inferred)
    assert tpr == 1.0
    np.testing.assert_allclose(fpr, 2 / (2 * n_nonlinked_pairs + 16))  # the diagonal adds 16 true negatives


@pytest.mark.parametrize("which", ["true", "inferred"])
def test_tpr_fpr_rejects_weighted_networks(which):
    binary = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weighted = np.array([[0, 1, 0], [1, 0, 0.4], [0, 0.4, 0]])
    args = (weighted, binary) if which == "true" else (binary, weighted)
    with pytest.raises(ValueError, match="must be binary"):
        tpr_fpr(*args)


def test_tpr_fpr_rejects_matrices_of_different_shapes():
    with pytest.raises(ValueError, match="different shapes"):
        tpr_fpr(np.zeros((3, 3)), np.zeros((4, 4)))


@pytest.mark.parametrize("shape", [(200,), (200, 1)])
def test_mir_surrogate_test_needs_two_nodes(shape):
    with pytest.raises(ValueError, match="at least two nodes"):
        mir_surrogate_test(np.random.default_rng(0).random(shape), p_sl=0.5)


def test_mir_surrogate_test_checks_the_true_network_first():
    with pytest.raises(ValueError, match="must be binary"):
        mir_surrogate_test(np.random.default_rng(0).random((200, 3)), p_sl=0.5,
                           true_network=np.full((3, 3), 0.5))


def test_with_ten_surrogates_a_link_must_beat_every_surrogate():
    # p-values are multiples of 1/10; with q = p_sl / 100 = 0.001 only p = 0 passes
    p_values = np.array([0.0, 0.0, 0.1, 0.2, 0.5, 1.0])
    rejected, _, _, _ = fdr_bh(p_values, 0.001)
    np.testing.assert_array_equal(rejected, p_values == 0)
