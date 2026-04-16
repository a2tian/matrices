from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from ..operators import PSDOperator, paired_entries
from ..results import ApproximationResult
from .base import ApproximationMethod, current_entry_count, validate_rank

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.int_]


class _PivotSelectorContext(Protocol):
    diagonal: FloatArray

    def entry(self, i: int, j: int) -> float: ...

    def entries(self, rows: NDArray[np.int_], cols: NDArray[np.int_]) -> FloatArray: ...


PivotSelector = Callable[[_PivotSelectorContext, Generator], int]


@dataclass(slots=True)
class _ResidualSelectorContext:
    operator: PSDOperator
    factors: FloatArray
    diagonal: FloatArray

    def entries(self, rows: NDArray[np.int_], cols: NDArray[np.int_]) -> FloatArray:
        row_indices = np.asarray(rows, dtype=int)
        col_indices = np.asarray(cols, dtype=int)
        if row_indices.ndim != 1 or col_indices.ndim != 1:
            raise ValueError("rows and cols must be one-dimensional")
        if row_indices.shape[0] != col_indices.shape[0]:
            raise ValueError("rows and cols must have the same length")
        base_entries = paired_entries(self.operator, row_indices.tolist(), col_indices.tolist())
        if self.factors.shape[1] == 0 or row_indices.size == 0:
            return base_entries
        correction = np.sum(
            self.factors[row_indices, :] * self.factors[col_indices, :],
            axis=1,
        )
        return np.asarray(base_entries - correction, dtype=float)

    def entry(self, i: int, j: int) -> float:
        return float(
            self.entries(
                np.array([i], dtype=int),
                np.array([j], dtype=int),
            )[0]
        )


@dataclass(slots=True)
class _AliasSampler:
    thresholds: FloatArray
    aliases: IndexArray

    @classmethod
    def from_probabilities(cls, probabilities: FloatArray) -> _AliasSampler:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1:
            raise ValueError("probabilities must be one-dimensional")
        if values.size == 0:
            raise ValueError("probabilities must be non-empty")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("probabilities must have positive mass")

        normalized = values / total
        scaled = normalized * normalized.size
        thresholds = np.ones(normalized.size, dtype=float)
        aliases = np.arange(normalized.size, dtype=int)

        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]

        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            thresholds[small_index] = float(scaled[small_index])
            aliases[small_index] = large_index
            scaled[large_index] = scaled[large_index] - (1.0 - thresholds[small_index])
            if scaled[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)

        for index in small + large:
            thresholds[index] = 1.0
            aliases[index] = index

        return cls(
            thresholds=np.asarray(np.clip(thresholds, 0.0, 1.0), dtype=float),
            aliases=np.asarray(aliases, dtype=int),
        )

    def sample(self, size: int, rng: Generator) -> IndexArray:
        sample_size = int(size)
        if sample_size < 0:
            raise ValueError("size must be nonnegative")
        if sample_size == 0:
            return np.zeros(0, dtype=int)
        primary = np.asarray(rng.integers(self.aliases.size, size=sample_size), dtype=int)
        uniforms = rng.random(sample_size)
        return np.where(uniforms < self.thresholds[primary], primary, self.aliases[primary])


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

    sampler = _AliasSampler.from_probabilities(probabilities)
    n_items = len(context.diagonal)
    batch_size = 1

    while True:
        rows = sampler.sample(batch_size, rng)
        cols = sampler.sample(batch_size, rng)

        denom = context.diagonal[rows] * context.diagonal[cols]
        accept_probabilities = np.zeros(batch_size, dtype=float)
        valid = denom > 0.0
        if np.any(valid):
            residual_entries = context.entries(rows[valid], cols[valid])
            accept_probabilities[valid] = np.clip(
                (residual_entries * residual_entries) / denom[valid],
                0.0,
                1.0,
            )

        accepted = rng.random(batch_size) <= accept_probabilities
        if np.any(accepted):
            first_accepted = int(np.flatnonzero(accepted)[0])
            return int(rows[first_accepted])

        batch_size = min(2 * batch_size, n_items)


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
