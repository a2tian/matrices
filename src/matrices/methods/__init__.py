from __future__ import annotations

from .base import ApproximationMethod
from .cholesky import (
    ColumnNormCholeskyMethod,
    GreedyCholeskyMethod,
    RandomPivotedCholeskyMethod,
)
from .nystrom import DiagonalWeightedNystromMethod, UniformNystromMethod
from .projector import (
    ProjectedApproximationMethod,
    ProjectedColumnNormCholeskyMethod,
    ProjectedDiagonalWeightedNystromMethod,
    ProjectedGreedyCholeskyMethod,
    ProjectedRandomPivotedCholeskyMethod,
    ProjectedUniformNystromMethod,
)

BUILTIN_METHODS: dict[str, type[ApproximationMethod]] = {
    "uniform_nystrom": UniformNystromMethod,
    "diagonal_nystrom": DiagonalWeightedNystromMethod,
    "column_norm_cholesky": ColumnNormCholeskyMethod,
    "greedy_cholesky": GreedyCholeskyMethod,
    "projected_column_norm_cholesky": ProjectedColumnNormCholeskyMethod,
    "projected_diagonal_nystrom": ProjectedDiagonalWeightedNystromMethod,
    "projected_greedy_cholesky": ProjectedGreedyCholeskyMethod,
    "projected_rp_cholesky": ProjectedRandomPivotedCholeskyMethod,
    "projected_uniform_nystrom": ProjectedUniformNystromMethod,
    "rp_cholesky": RandomPivotedCholeskyMethod,
}


def build_method(name: str) -> ApproximationMethod:
    try:
        return BUILTIN_METHODS[name]()
    except KeyError as exc:
        message = f"unknown method '{name}', expected one of {sorted(BUILTIN_METHODS)}"
        raise KeyError(message) from exc


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_METHODS))


__all__ = [
    "ApproximationMethod",
    "BUILTIN_METHODS",
    "DiagonalWeightedNystromMethod",
    "ColumnNormCholeskyMethod",
    "GreedyCholeskyMethod",
    "ProjectedApproximationMethod",
    "ProjectedColumnNormCholeskyMethod",
    "ProjectedDiagonalWeightedNystromMethod",
    "ProjectedGreedyCholeskyMethod",
    "ProjectedRandomPivotedCholeskyMethod",
    "ProjectedUniformNystromMethod",
    "RandomPivotedCholeskyMethod",
    "UniformNystromMethod",
    "available_methods",
    "build_method",
]
