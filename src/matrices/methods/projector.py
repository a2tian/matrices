from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.random import Generator

from ..numerics import orthonormal_column_basis, psd_eigendecomposition, symmetrize
from ..operators import PSDOperator, apply_operator
from ..results import ApproximationResult
from .base import ApproximationMethod, current_entry_count
from .cholesky import ColumnNormCholeskyMethod, GreedyCholeskyMethod, RandomPivotedCholeskyMethod
from .nystrom import DiagonalWeightedNystromMethod, UniformNystromMethod


def _unique_indices(indices: tuple[int, ...]) -> tuple[int, ...]:
    unique: list[int] = []
    seen: set[int] = set()
    for index in indices:
        scalar_index = int(index)
        if scalar_index in seen:
            continue
        unique.append(scalar_index)
        seen.add(scalar_index)
    return tuple(unique)


@dataclass(slots=True)
class ProjectedApproximationMethod(ApproximationMethod):
    base_method: ApproximationMethod
    name: str = ""
    rcond: float = 1e-10
    deterministic: bool = field(init=False)

    def __post_init__(self) -> None:
        self.deterministic = bool(self.base_method.deterministic)
        if not self.name:
            self.name = f"projected_{self.base_method.name}"

    def run(self, operator: PSDOperator, rank: int, rng: Generator) -> ApproximationResult:
        start = perf_counter()
        base_result = self.base_method.run(operator, rank, rng)
        selected_indices = _unique_indices(base_result.selected_indices)
        n_rows = operator.shape[0]

        if not selected_indices:
            factors = np.zeros((n_rows, 0), dtype=float)
            metadata = {
                "construction": "orthogonal_projector",
                "base_method": base_result.method,
                "selected_count": 0,
                "projector_basis_rank": 0,
            }
            return ApproximationResult(
                method=self.name,
                selected_indices=selected_indices,
                factors=factors,
                effective_rank=0,
                runtime_seconds=perf_counter() - start,
                entry_evaluations=current_entry_count(operator),
                metadata=metadata,
            )

        row_indices = list(range(n_rows))
        columns = operator.submatrix(row_indices, selected_indices)
        basis = orthonormal_column_basis(columns, rcond=self.rcond)

        if basis.shape[1] == 0:
            factors = np.zeros((n_rows, 0), dtype=float)
        else:
            applied_basis = apply_operator(operator, basis)
            compressed = symmetrize(basis.T @ applied_basis)
            eigenvalues, eigenvectors = psd_eigendecomposition(compressed)
            max_eigenvalue = float(np.max(np.abs(eigenvalues), initial=0.0))
            threshold = self.rcond * max(max_eigenvalue, 1.0)
            keep = eigenvalues > threshold
            if not np.any(keep):
                factors = np.zeros((n_rows, 0), dtype=float)
            else:
                factors = np.asarray(
                    basis @ (eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])),
                    dtype=float,
                )

        metadata = {
            "construction": "orthogonal_projector",
            "base_method": base_result.method,
            "selected_count": len(selected_indices),
            "projector_basis_rank": int(basis.shape[1]),
        }
        return ApproximationResult(
            method=self.name,
            selected_indices=selected_indices,
            factors=factors,
            effective_rank=int(factors.shape[1]),
            runtime_seconds=perf_counter() - start,
            entry_evaluations=current_entry_count(operator),
            metadata=metadata,
        )


@dataclass(slots=True)
class ProjectedUniformNystromMethod(ProjectedApproximationMethod):
    base_method: ApproximationMethod = field(default_factory=UniformNystromMethod)
    name: str = "projected_uniform_nystrom"


@dataclass(slots=True)
class ProjectedDiagonalWeightedNystromMethod(ProjectedApproximationMethod):
    base_method: ApproximationMethod = field(default_factory=DiagonalWeightedNystromMethod)
    name: str = "projected_diagonal_nystrom"


@dataclass(slots=True)
class ProjectedGreedyCholeskyMethod(ProjectedApproximationMethod):
    base_method: ApproximationMethod = field(default_factory=GreedyCholeskyMethod)
    name: str = "projected_greedy_cholesky"


@dataclass(slots=True)
class ProjectedRandomPivotedCholeskyMethod(ProjectedApproximationMethod):
    base_method: ApproximationMethod = field(default_factory=RandomPivotedCholeskyMethod)
    name: str = "projected_rp_cholesky"


@dataclass(slots=True)
class ProjectedColumnNormCholeskyMethod(ProjectedApproximationMethod):
    base_method: ApproximationMethod = field(default_factory=ColumnNormCholeskyMethod)
    name: str = "projected_column_norm_cholesky"
