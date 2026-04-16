from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from matrices.methods import (
    ApproximationMethod,
    ColumnNormCholeskyMethod,
    DiagonalWeightedNystromMethod,
    GreedyCholeskyMethod,
    ProjectedApproximationMethod,
    ProjectedColumnNormCholeskyMethod,
    ProjectedDiagonalWeightedNystromMethod,
    ProjectedGreedyCholeskyMethod,
    ProjectedRandomPivotedCholeskyMethod,
    ProjectedUniformNystromMethod,
    RandomPivotedCholeskyMethod,
    UniformNystromMethod,
    available_methods,
)
from matrices.numerics import nystrom_factor
from matrices.operators import CountingPSDOperator, DensePSDOperator, PSDOperator
from matrices.results import ApproximationResult

FloatArray: TypeAlias = NDArray[np.float64]


def _random_psd_matrix(seed: int, n_rows: int = 20, n_features: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_rows, n_features))
    return matrix @ matrix.T


@dataclass(slots=True)
class _FixedSelectionMethod(ApproximationMethod):
    selected_indices: tuple[int, ...]
    name: str = "fixed_selection"
    deterministic: bool = True

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        del rank, rng
        return ApproximationResult(
            method=self.name,
            selected_indices=self.selected_indices,
            factors=np.zeros((operator.shape[0], 0), dtype=float),
        )


@dataclass(slots=True)
class _BatchedEntryOnlyRankOneOperator:
    matrix: FloatArray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def diagonal(self) -> FloatArray:
        return np.diag(self.matrix).copy()

    def entry(self, row: int, col: int) -> float:
        raise AssertionError("column-norm selector should use batched entries")

    def entries(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        row_indices = np.asarray(rows, dtype=int)
        col_indices = np.asarray(cols, dtype=int)
        return np.asarray(self.matrix[row_indices, col_indices], dtype=float)

    def column(self, index: int) -> FloatArray:
        return self.matrix[:, index].copy()

    def submatrix(self, rows: Sequence[int], cols: Sequence[int]) -> FloatArray:
        row_indices = np.asarray(rows, dtype=int)
        col_indices = np.asarray(cols, dtype=int)
        return np.asarray(self.matrix[np.ix_(row_indices, col_indices)], dtype=float)

    def materialize(self) -> FloatArray:
        return self.matrix.copy()


def test_methods_produce_symmetric_approximations() -> None:
    operator = DensePSDOperator(_random_psd_matrix(4))
    methods = [
        UniformNystromMethod(),
        DiagonalWeightedNystromMethod(),
        ColumnNormCholeskyMethod(),
        GreedyCholeskyMethod(),
        ProjectedColumnNormCholeskyMethod(),
        ProjectedUniformNystromMethod(),
        ProjectedDiagonalWeightedNystromMethod(),
        ProjectedGreedyCholeskyMethod(),
        ProjectedRandomPivotedCholeskyMethod(),
        RandomPivotedCholeskyMethod(),
    ]

    for method in methods:
        result = method.run(operator, rank=5, rng=np.random.default_rng(3))
        approximation = result.materialize()
        assert result.effective_rank <= 5
        assert approximation.shape == operator.shape
        assert np.allclose(approximation, approximation.T, atol=1e-10)


def test_random_pivoted_cholesky_matches_nystrom_identity() -> None:
    matrix = _random_psd_matrix(9, n_rows=24, n_features=7)
    operator = DensePSDOperator(matrix)
    result = RandomPivotedCholeskyMethod().run(operator, rank=6, rng=np.random.default_rng(5))

    indices = list(result.selected_indices)
    basis = operator.submatrix(list(range(operator.shape[0])), indices)
    intersection = operator.submatrix(indices, indices)
    nystrom = nystrom_factor(basis, intersection)

    assert np.allclose(result.materialize(), nystrom @ nystrom.T, atol=1e-8)


def test_method_registry_contains_expected_names() -> None:
    assert available_methods() == (
        "column_norm_cholesky",
        "diagonal_nystrom",
        "greedy_cholesky",
        "projected_column_norm_cholesky",
        "projected_diagonal_nystrom",
        "projected_greedy_cholesky",
        "projected_rp_cholesky",
        "projected_uniform_nystrom",
        "rp_cholesky",
        "uniform_nystrom",
    )


def test_column_norm_cholesky_reconstructs_rank_one_with_lazy_entry_queries() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.outer(u, u)
    operator = CountingPSDOperator(DensePSDOperator(matrix))

    result = ColumnNormCholeskyMethod().run(operator, rank=1, rng=np.random.default_rng(7))

    assert result.effective_rank == 1
    assert np.allclose(result.materialize(), matrix, atol=1e-10)
    assert result.entry_evaluations == matrix.shape[0] + 1 + matrix.shape[0]
    assert operator.entry_evaluations == matrix.shape[0] + 1 + matrix.shape[0]


def test_projected_method_is_exact_for_diagonal_selection() -> None:
    matrix = np.diag([2.0, 3.0, 5.0, 7.0])
    operator = DensePSDOperator(matrix)
    method = ProjectedApproximationMethod(_FixedSelectionMethod((0, 2)))

    result = method.run(operator, rank=2, rng=np.random.default_rng(0))

    expected = np.diag([2.0, 0.0, 5.0, 0.0])
    assert result.selected_indices == (0, 2)
    assert result.effective_rank == 2
    assert np.allclose(result.materialize(), expected, atol=1e-10)


def test_projected_method_handles_duplicate_or_dependent_columns() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.outer(u, u)
    operator = DensePSDOperator(matrix)
    method = ProjectedApproximationMethod(_FixedSelectionMethod((0, 0, 2)))

    result = method.run(operator, rank=3, rng=np.random.default_rng(0))

    assert result.selected_indices == (0, 2)
    assert result.effective_rank == 1
    assert np.allclose(result.materialize(), matrix, atol=1e-10)


def test_column_norm_cholesky_uses_batched_entries_in_selector() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.outer(u, u)
    operator = _BatchedEntryOnlyRankOneOperator(matrix)

    result = ColumnNormCholeskyMethod().run(operator, rank=1, rng=np.random.default_rng(0))

    assert result.effective_rank == 1
    assert np.allclose(result.materialize(), matrix, atol=1e-10)
