from __future__ import annotations

import numpy as np

from matrices.methods import (
    DiagonalWeightedNystromMethod,
    ExactColumnNormCholeskyMethod,
    GreedyCholeskyMethod,
    RandomPivotedCholeskyMethod,
    UniformNystromMethod,
    available_methods,
)
from matrices.numerics import nystrom_factor
from matrices.operators import CountingPSDOperator, DensePSDOperator


def _random_psd_matrix(seed: int, n_rows: int = 20, n_features: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_rows, n_features))
    return matrix @ matrix.T


def test_methods_produce_symmetric_approximations() -> None:
    operator = DensePSDOperator(_random_psd_matrix(4))
    methods = [
        UniformNystromMethod(),
        DiagonalWeightedNystromMethod(),
        ExactColumnNormCholeskyMethod(),
        GreedyCholeskyMethod(),
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

    indices = np.array(result.selected_indices, dtype=int)
    basis = operator.submatrix(np.arange(operator.shape[0]), indices)
    intersection = operator.submatrix(indices, indices)
    nystrom = nystrom_factor(basis, intersection)

    assert np.allclose(result.materialize(), nystrom @ nystrom.T, atol=1e-8)


def test_method_registry_contains_expected_names() -> None:
    assert available_methods() == (
        "diagonal_nystrom",
        "exact_column_norm_cholesky",
        "greedy_cholesky",
        "rp_cholesky",
        "uniform_nystrom",
    )


def test_exact_column_norm_cholesky_reconstructs_rank_one_with_lazy_entry_queries() -> None:
    u = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.outer(u, u)
    operator = CountingPSDOperator(DensePSDOperator(matrix))

    result = ExactColumnNormCholeskyMethod().run(operator, rank=1, rng=np.random.default_rng(7))

    assert result.effective_rank == 1
    assert np.allclose(result.materialize(), matrix, atol=1e-10)
    assert result.entry_evaluations == matrix.shape[0] + 1 + matrix.shape[0]
    assert operator.entry_evaluations == matrix.shape[0] + 1 + matrix.shape[0]
