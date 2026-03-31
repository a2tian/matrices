from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.random import Generator

from ..operators import PSDOperator
from ..results import ApproximationResult


class ApproximationMethod(ABC):
    """Base class for all approximation methods."""

    name: str
    deterministic: bool = False

    @abstractmethod
    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        """Approximate an operator with a rank-constrained model."""

    def __call__(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        return self.run(operator, rank, rng)


def validate_rank(operator: PSDOperator, rank: int) -> int:
    if rank <= 0:
        raise ValueError("rank must be positive")
    return min(rank, operator.shape[0])


def current_entry_count(operator: PSDOperator) -> int:
    return int(getattr(operator, "entry_evaluations", 0))
