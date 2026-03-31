from __future__ import annotations

import numpy as np

from matrices.kernels import GaussianKernel
from matrices.operators import CountingPSDOperator, DensePSDOperator, KernelPSDOperator


def test_dense_operator_accessors_match_source_matrix() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    operator = DensePSDOperator(matrix)

    assert operator.shape == (2, 2)
    assert np.allclose(operator.diagonal(), np.array([2.0, 3.0]))
    assert np.allclose(operator.column(0), np.array([2.0, 1.0]))
    assert np.allclose(operator.submatrix([0, 1], [1]), np.array([[1.0], [3.0]]))
    assert np.allclose(operator.materialize(), matrix)


def test_kernel_operator_matches_dense_kernel_matrix() -> None:
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    kernel = GaussianKernel(bandwidth=1.5)
    operator = KernelPSDOperator(data, kernel)

    dense = kernel.matrix(data, data)
    assert np.allclose(operator.diagonal(), np.diag(dense))
    assert np.allclose(operator.column(1), dense[:, 1])
    assert np.allclose(operator.submatrix([0, 2], [1, 2]), dense[np.ix_([0, 2], [1, 2])])
    assert np.allclose(operator.materialize(), dense)


def test_counting_operator_tracks_entry_queries() -> None:
    operator = CountingPSDOperator(DensePSDOperator(np.eye(4)))
    operator.diagonal()
    operator.column(0)
    operator.submatrix([0, 1], [2, 3])

    assert operator.entry_evaluations == 4 + 4 + 4
