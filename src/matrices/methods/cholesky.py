from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.random import Generator

from ..operators import PSDOperator
from ..results import ApproximationResult
from .base import ApproximationMethod, current_entry_count, validate_rank


def _greedy_selector(diagonal: np.ndarray, _: Generator) -> int:
    return int(np.argmax(diagonal))


def _random_selector(diagonal: np.ndarray, rng: Generator) -> int:
    weights = np.clip(diagonal, a_min=0.0, a_max=None)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return int(np.argmax(diagonal))
    probabilities = weights / total_weight
    return int(rng.choice(len(diagonal), p=probabilities))


@dataclass(slots=True)
class _PartialCholeskyMethod(ApproximationMethod):
    name: str = ""
    deterministic: bool = False
    selector: Callable[[np.ndarray, Generator], int] = _greedy_selector

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
            pivot_index = int(self.selector(diagonal, rng))
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
    selector: Callable[[np.ndarray, Generator], int] = _greedy_selector


@dataclass(slots=True)
class RandomPivotedCholeskyMethod(_PartialCholeskyMethod):
    name: str = "rp_cholesky"
    deterministic: bool = False
    selector: Callable[[np.ndarray, Generator], int] = _random_selector
