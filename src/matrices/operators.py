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

    def entry(self, row: int, col: int) -> float:
        """Return a single operator entry."""

    def column(self, index: int) -> FloatArray:
        """Return a dense view of the requested column."""

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        """Return a dense submatrix for the given row and column indices."""

    def materialize(self) -> FloatArray:
        """Return the dense matrix representation."""


def apply_operator(operator: PSDOperator, vectors: FloatArray) -> FloatArray:
    """Return operator @ vectors without requiring materialization."""

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    if matrix.shape[0] != operator.shape[1]:
        raise ValueError("vectors must have shape (operator.shape[1], k)")

    fast_apply = getattr(operator, "apply", None)
    if callable(fast_apply):
        return np.asarray(fast_apply(matrix), dtype=float)

    result = np.zeros((operator.shape[0], matrix.shape[1]), dtype=float)
    for column_index in range(operator.shape[1]):
        coefficients = matrix[column_index, :]
        if not np.any(coefficients):
            continue
        column = operator.column(column_index)
        result += np.outer(column, coefficients)
    return result


def _as_index_array(indices: Sequence[int], limit: int) -> IndexArray:
    array = np.asarray(indices, dtype=int)
    if array.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if np.any(array < 0) or np.any(array >= limit):
        raise IndexError("index out of bounds")
    return array


def _as_scalar_index(index: int, limit: int) -> int:
    scalar = int(index)
    if scalar < 0 or scalar >= limit:
        raise IndexError("index out of bounds")
    return scalar


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

    def entry(self, row: int, col: int) -> float:
        row_index = _as_scalar_index(row, self.shape[0])
        col_index = _as_scalar_index(col, self.shape[1])
        return float(self.matrix[row_index, col_index])

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

    def entry(self, row: int, col: int) -> float:
        row_index = _as_scalar_index(row, self.shape[0])
        col_index = _as_scalar_index(col, self.shape[1])
        return float(self.kernel.pair(self.data[row_index], self.data[col_index]))

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

    def entry(self, row: int, col: int) -> float:
        value = self.base.entry(row, col)
        self.entry_evaluations += 1
        return value

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
