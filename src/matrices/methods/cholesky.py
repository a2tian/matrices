from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ..operators import PSDOperator
from ..results import ApproximationResult
from .base import ApproximationMethod, current_entry_count, validate_rank

FloatArray: TypeAlias = NDArray[np.float64]


class _PivotSelectorContext(Protocol):
    diagonal: FloatArray

    def entry(self, i: int, j: int) -> float: ...


PivotSelector = Callable[[_PivotSelectorContext, Generator], int]


@dataclass(slots=True)
class _ResidualSelectorContext:
    operator: PSDOperator
    factors: FloatArray
    diagonal: FloatArray

    def entry(self, i: int, j: int) -> float:
        correction = 0.0
        if self.factors.shape[1] > 0:
            correction = float(self.factors[i, :] @ self.factors[j, :])
        return float(self.operator.entry(i, j) - correction)


def _diagonal_sampling_probabilities(diagonal: FloatArray) -> FloatArray | None:
    weights: FloatArray = np.clip(np.asarray(diagonal, dtype=float), a_min=0.0, a_max=None)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return None
    probabilities: FloatArray = weights / total_weight
    return probabilities


def _greedy_selector(context: _PivotSelectorContext, _: Generator) -> int:
    return int(np.argmax(context.diagonal))


def _random_selector(context: _PivotSelectorContext, rng: Generator) -> int:
    probabilities = _diagonal_sampling_probabilities(context.diagonal)
    if probabilities is None:
        return int(np.argmax(context.diagonal))
    return int(rng.choice(len(context.diagonal), p=probabilities))


def _column_norm_selector(context: _PivotSelectorContext, rng: Generator) -> int:
    probabilities = _diagonal_sampling_probabilities(context.diagonal)
    if probabilities is None:
        return int(np.argmax(context.diagonal))

    while True:
        i = int(rng.choice(len(context.diagonal), p=probabilities))
        j = int(rng.choice(len(context.diagonal), p=probabilities))
        d_i = float(context.diagonal[i])
        d_j = float(context.diagonal[j])
        if d_i <= 0 or d_j <= 0:
            continue
        r_ij = float(context.entry(i, j))
        accept_probability = float(np.clip((r_ij * r_ij) / (d_i * d_j), 0.0, 1.0))
        if rng.random() <= accept_probability:
            return i


@dataclass(slots=True)
class _PartialCholeskyMethod(ApproximationMethod):
    name: str = ""
    deterministic: bool = False
    selector: PivotSelector = _greedy_selector

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        target_rank = validate_rank(operator, rank)
        n_rows = operator.shape[0]
        start = perf_counter()
        diagonal = np.clip(operator.diagonal().astype(float), a_min=0.0, a_max=None)
        factors = np.zeros((n_rows, target_rank), dtype=float)
        selected: list[int] = []

        for column_index in range(target_rank):
            if float(diagonal.sum()) <= 0:
                break
            context = _ResidualSelectorContext(
                operator=operator,
                factors=factors[:, :column_index],
                diagonal=diagonal,
            )
            pivot_index = int(self.selector(context, rng))
            pivot_value = float(diagonal[pivot_index])
            if pivot_value <= 0:
                break
            column = operator.column(pivot_index)
            if column_index == 0:
                residual = column
            else:
                residual = column - factors[:, :column_index] @ factors[pivot_index, :column_index]
            pivot = float(residual[pivot_index])
            if pivot <= 1e-12:
                break
            factors[:, column_index] = residual / np.sqrt(pivot)
            diagonal = np.clip(diagonal - factors[:, column_index] ** 2, a_min=0.0, a_max=None)
            selected.append(pivot_index)

        runtime_seconds = perf_counter() - start
        used_factors = factors[:, : len(selected)]
        return ApproximationResult(
            method=self.name,
            selected_indices=tuple(selected),
            factors=used_factors,
            effective_rank=used_factors.shape[1],
            runtime_seconds=runtime_seconds,
            entry_evaluations=current_entry_count(operator),
            metadata={"factorization": "partial_cholesky"},
        )


@dataclass(slots=True)
class GreedyCholeskyMethod(_PartialCholeskyMethod):
    name: str = "greedy_cholesky"
    deterministic: bool = True
    selector: PivotSelector = _greedy_selector


@dataclass(slots=True)
class RandomPivotedCholeskyMethod(_PartialCholeskyMethod):
    name: str = "rp_cholesky"
    deterministic: bool = False
    selector: PivotSelector = _random_selector


@dataclass(slots=True)
class ColumnNormCholeskyMethod(_PartialCholeskyMethod):
    name: str = "column_norm_cholesky"
    deterministic: bool = False
    selector: PivotSelector = _column_norm_selector
