from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class KernelFunction(Protocol):
    """Matrix-valued kernel interface used by lazy PSD operators."""

    def pair(self, x: FloatArray, y: FloatArray) -> float:
        """Evaluate the kernel on a pair of points."""

    def matrix(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Evaluate the kernel on two collections of points."""


@dataclass(slots=True, frozen=True)
class GaussianKernel:
    """Gaussian kernel with scalar bandwidth."""

    bandwidth: float = 1.0

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")

    def pair(self, x: FloatArray, y: FloatArray) -> float:
        diff = x - y
        scale = -0.5 / (self.bandwidth**2)
        return float(np.exp(scale * np.dot(diff, diff)))

    def matrix(self, x: FloatArray, y: FloatArray) -> FloatArray:
        x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y**2, axis=1)
        distances = x_norm - 2.0 * x @ y.T + y_norm
        return np.asarray(np.exp((-0.5 / (self.bandwidth**2)) * distances), dtype=float)

    def __call__(self, x: FloatArray, y: FloatArray) -> float:
        return self.pair(x, y)
