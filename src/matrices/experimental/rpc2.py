from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.random import Generator

from ..methods.base import ApproximationMethod, current_entry_count, validate_rank
from ..operators import PSDOperator
from ..results import ApproximationResult


def best_column2(matrix: np.ndarray) -> int:
    gram = matrix.T @ matrix
    diagonal = np.diag(gram)
    safe_diagonal = np.where(diagonal > 0, diagonal, np.inf)
    scores = np.diag(gram @ gram) / safe_diagonal
    return int(np.argmax(scores))


@dataclass(slots=True)
class RPC2CholeskyMethod(ApproximationMethod):
    """Experimental selector based on the original research prototype."""

    name: str = "rpc2_cholesky"
    deterministic: bool = False

    def _select(
        self,
        operator: PSDOperator,
        diagonal: np.ndarray,
        factors: np.ndarray,
        rng: Generator,
    ) -> int:
        total_weight = float(diagonal.sum())
        if total_weight <= 0:
            return int(np.argmax(diagonal))
        probabilities = diagonal / total_weight
        sample_size = max(1, math.isqrt(operator.shape[0]))
        sample_size = min(sample_size, operator.shape[0])
        subset = rng.choice(operator.shape[0], size=sample_size, replace=False, p=probabilities)
        subset_list = subset.tolist()
        residual = (
            operator.submatrix(subset_list, subset_list) - factors[subset, :] @ factors[subset, :].T
        )
        scaling = np.diag(1.0 / np.sqrt(probabilities[subset]))
        score_matrix = scaling @ residual @ scaling
        return int(subset[best_column2(score_matrix)])

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        target_rank = validate_rank(operator, rank)
        n_rows = operator.shape[0]
        diagonal = np.clip(operator.diagonal().astype(float), a_min=0.0, a_max=None)
        factors = np.zeros((n_rows, target_rank), dtype=float)
        selected: list[int] = []
        start = perf_counter()

        for column_index in range(target_rank):
            if float(diagonal.sum()) <= 0:
                break
            pivot_index = self._select(operator, diagonal, factors[:, :column_index], rng)
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
            metadata={"factorization": "partial_cholesky", "experimental": True},
        )
