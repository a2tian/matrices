from __future__ import annotations

import numpy as np
import pytest

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


def test_dense_operator_entry_matches_source_matrix() -> None:
    matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
    operator = DensePSDOperator(matrix)

    assert operator.entry(0, 1) == pytest.approx(matrix[0, 1])
    assert operator.entry(1, 0) == pytest.approx(matrix[1, 0])


def test_kernel_operator_matches_dense_kernel_matrix() -> None:
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    kernel = GaussianKernel(bandwidth=1.5)
    operator = KernelPSDOperator(data, kernel)

    dense = kernel.matrix(data, data)
    assert np.allclose(operator.diagonal(), np.diag(dense))
    assert np.allclose(operator.column(1), dense[:, 1])
    assert np.allclose(operator.submatrix([0, 2], [1, 2]), dense[np.ix_([0, 2], [1, 2])])
    assert np.allclose(operator.materialize(), dense)


def test_kernel_operator_entry_matches_dense_kernel_matrix() -> None:
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    kernel = GaussianKernel(bandwidth=1.5)
    operator = KernelPSDOperator(data, kernel)

    dense = kernel.matrix(data, data)
    assert operator.entry(2, 1) == pytest.approx(dense[2, 1])


def test_counting_operator_tracks_entry_queries() -> None:
    operator = CountingPSDOperator(DensePSDOperator(np.eye(4)))
    operator.diagonal()
    operator.column(0)
    operator.submatrix([0, 1], [2, 3])

    assert operator.entry_evaluations == 4 + 4 + 4


def test_counting_operator_counts_scalar_entry_queries() -> None:
    operator = CountingPSDOperator(DensePSDOperator(np.eye(3)))

    assert operator.entry(1, 1) == pytest.approx(1.0)
    assert operator.entry_evaluations == 1


@pytest.mark.parametrize("row,col", [(-1, 0), (0, 2)])
def test_entry_raises_index_error_out_of_bounds(row: int, col: int) -> None:
    operator = DensePSDOperator(np.eye(2))

    with pytest.raises(IndexError):
        operator.entry(row, col)
