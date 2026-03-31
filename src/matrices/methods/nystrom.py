from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.random import Generator

from ..numerics import nystrom_factor
from ..operators import PSDOperator
from ..results import ApproximationResult
from .base import ApproximationMethod, current_entry_count, validate_rank


def _sample_without_replacement(
    rng: Generator,
    n_items: int,
    size: int,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.arange(n_items, dtype=int)
    sample_size = min(size, n_items)
    if sample_size <= 0:
        return np.zeros(0, dtype=int)

    if weights is None:
        return np.sort(rng.choice(candidates, size=sample_size, replace=False))

    local_candidates = candidates.copy()
    local_weights = np.clip(np.asarray(weights, dtype=float), a_min=0.0, a_max=None).copy()
    chosen: list[int] = []
    for _ in range(sample_size):
        if local_candidates.size == 0:
            break
        if float(local_weights.sum()) <= 0:
            position = int(rng.integers(local_candidates.size))
        else:
            position = int(rng.choice(local_candidates.size, p=local_weights / local_weights.sum()))
        chosen.append(int(local_candidates[position]))
        local_candidates = np.delete(local_candidates, position)
        local_weights = np.delete(local_weights, position)
    return np.asarray(sorted(chosen), dtype=int)


def _build_nystrom_result(
    operator: PSDOperator,
    *,
    method_name: str,
    indices: np.ndarray,
    metadata: dict[str, object],
) -> ApproximationResult:
    row_indices = list(range(operator.shape[0]))
    selected_indices = indices.tolist()
    basis = operator.submatrix(row_indices, selected_indices)
    intersection = operator.submatrix(selected_indices, selected_indices)
    factors = nystrom_factor(basis, intersection)
    return ApproximationResult(
        method=method_name,
        selected_indices=tuple(int(index) for index in selected_indices),
        basis=basis,
        intersection=intersection,
        factors=factors,
        effective_rank=factors.shape[1],
        entry_evaluations=current_entry_count(operator),
        metadata=metadata,
    )


@dataclass(slots=True)
class UniformNystromMethod(ApproximationMethod):
    name: str = "uniform_nystrom"
    deterministic: bool = False

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        target_rank = validate_rank(operator, rank)
        start = perf_counter()
        indices = _sample_without_replacement(rng, operator.shape[0], target_rank)
        result = _build_nystrom_result(
            operator,
            method_name=self.name,
            indices=indices,
            metadata={"sampling": "uniform"},
        )
        result.runtime_seconds = perf_counter() - start
        return result


@dataclass(slots=True)
class DiagonalWeightedNystromMethod(ApproximationMethod):
    name: str = "diagonal_nystrom"
    deterministic: bool = False

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        target_rank = validate_rank(operator, rank)
        start = perf_counter()
        diagonal = operator.diagonal()
        indices = _sample_without_replacement(rng, operator.shape[0], target_rank, weights=diagonal)
        result = _build_nystrom_result(
            operator,
            method_name=self.name,
            indices=indices,
            metadata={"sampling": "diagonal_weighted"},
        )
        result.runtime_seconds = perf_counter() - start
        return result
