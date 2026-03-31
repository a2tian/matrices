from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .kernels import KernelFunction

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]


class PSDOperator(Protocol):
    """Minimal operator interface for PSD approximation algorithms."""

    @property
    def shape(self) -> tuple[int, int]:
        """Operator shape."""

    def diagonal(self) -> FloatArray:
        """Return the diagonal entries of the operator."""

    def column(self, index: int) -> FloatArray:
        """Return a dense view of the requested column."""

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        """Return a dense submatrix for the given row and column indices."""

    def materialize(self) -> FloatArray:
        """Return the dense matrix representation."""


def _as_index_array(indices: Sequence[int], limit: int) -> IndexArray:
    array = np.asarray(indices, dtype=int)
    if array.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if np.any(array < 0) or np.any(array >= limit):
        raise IndexError("index out of bounds")
    return array


@dataclass(slots=True)
class DensePSDOperator:
    """Explicit dense PSD matrix wrapper."""

    matrix: FloatArray
    validate: bool = True

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if self.validate and not np.allclose(matrix, matrix.T, atol=1e-10):
            raise ValueError("matrix must be symmetric")
        self.matrix = matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def diagonal(self) -> FloatArray:
        return np.diag(self.matrix).copy()

    def column(self, index: int) -> FloatArray:
        return self.matrix[:, index].copy()

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        row_indices = _as_index_array(rows, self.shape[0])
        col_indices = _as_index_array(cols, self.shape[1])
        return self.matrix[np.ix_(row_indices, col_indices)].copy()

    def materialize(self) -> FloatArray:
        return self.matrix.copy()


@dataclass(slots=True)
class KernelPSDOperator:
    """Lazy PSD operator backed by a kernel and a set of feature vectors."""

    data: FloatArray
    kernel: KernelFunction

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array")
        self.data = data

    @property
    def shape(self) -> tuple[int, int]:
        return (self.data.shape[0], self.data.shape[0])

    def diagonal(self) -> FloatArray:
        return np.fromiter(
            (self.kernel.pair(row, row) for row in self.data),
            dtype=float,
            count=self.data.shape[0],
        )

    def column(self, index: int) -> FloatArray:
        column = self.kernel.matrix(self.data, self.data[index : index + 1])
        return column[:, 0]

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        row_indices = _as_index_array(rows, self.shape[0])
        col_indices = _as_index_array(cols, self.shape[1])
        return self.kernel.matrix(self.data[row_indices], self.data[col_indices])

    def materialize(self) -> FloatArray:
        return self.kernel.matrix(self.data, self.data)


@dataclass(slots=True)
class CountingPSDOperator:
    """Operator wrapper that counts dense entry evaluations."""

    base: PSDOperator
    entry_evaluations: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    def diagonal(self) -> FloatArray:
        diagonal = self.base.diagonal()
        self.entry_evaluations += diagonal.size
        return diagonal

    def column(self, index: int) -> FloatArray:
        column = self.base.column(index)
        self.entry_evaluations += column.size
        return column

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        submatrix = self.base.submatrix(rows, cols)
        self.entry_evaluations += submatrix.size
        return submatrix

    def materialize(self) -> FloatArray:
        materialized = self.base.materialize()
        self.entry_evaluations += materialized.size
        return materialized
