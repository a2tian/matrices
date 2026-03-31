from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Metadata = dict[str, object]


@dataclass(slots=True)
class ApproximationResult:
    """Container for a low-rank approximation returned by a method."""

    method: str
    selected_indices: tuple[int, ...]
    factors: FloatArray
    basis: FloatArray | None = None
    intersection: FloatArray | None = None
    effective_rank: int = 0
    runtime_seconds: float = 0.0
    entry_evaluations: int = 0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        factors = np.asarray(self.factors, dtype=float)
        if factors.ndim != 2:
            raise ValueError("factors must be two-dimensional")
        self.factors = factors
        if self.basis is not None:
            self.basis = np.asarray(self.basis, dtype=float)
        if self.intersection is not None:
            self.intersection = np.asarray(self.intersection, dtype=float)
        if self.effective_rank <= 0:
            self.effective_rank = int(factors.shape[1])

    def materialize(self) -> FloatArray:
        return self.factors @ self.factors.T
