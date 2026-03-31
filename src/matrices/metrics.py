from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .numerics import symmetrize

FloatArray = NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class MetricSnapshot:
    relative_frobenius_error: float
    relative_trace_error: float


def _safe_relative_error(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return numerator / denominator


def relative_frobenius_error(target: FloatArray, approximation: FloatArray) -> float:
    residual = target - approximation
    numerator = float(np.linalg.norm(residual, ord="fro"))
    denominator = float(np.linalg.norm(target, ord="fro"))
    return _safe_relative_error(numerator, denominator)


def relative_trace_error(target: FloatArray, approximation: FloatArray) -> float:
    residual = symmetrize(target - approximation)
    numerator = max(float(np.trace(residual)), 0.0)
    denominator = float(np.trace(symmetrize(target)))
    return _safe_relative_error(numerator, denominator)


def evaluate_approximation(target: FloatArray, approximation: FloatArray) -> MetricSnapshot:
    sym_target = symmetrize(np.asarray(target, dtype=float))
    sym_approximation = symmetrize(np.asarray(approximation, dtype=float))
    return MetricSnapshot(
        relative_frobenius_error=relative_frobenius_error(sym_target, sym_approximation),
        relative_trace_error=relative_trace_error(sym_target, sym_approximation),
    )
