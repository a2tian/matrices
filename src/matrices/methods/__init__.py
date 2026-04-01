from __future__ import annotations

from .base import ApproximationMethod
from .cholesky import (
    ExactColumnNormCholeskyMethod,
    GreedyCholeskyMethod,
    RandomPivotedCholeskyMethod,
)
from .nystrom import DiagonalWeightedNystromMethod, UniformNystromMethod

BUILTIN_METHODS: dict[str, type[ApproximationMethod]] = {
    "uniform_nystrom": UniformNystromMethod,
    "diagonal_nystrom": DiagonalWeightedNystromMethod,
    "exact_column_norm_cholesky": ExactColumnNormCholeskyMethod,
    "greedy_cholesky": GreedyCholeskyMethod,
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
    "ExactColumnNormCholeskyMethod",
    "GreedyCholeskyMethod",
    "RandomPivotedCholeskyMethod",
    "UniformNystromMethod",
    "available_methods",
    "build_method",
]
