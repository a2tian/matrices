"""Reusable adaptive sampling methods for PSD matrix approximation."""

from .datasets import (
    CURATED_OPENML_DATASETS,
    DatasetBundle,
    DatasetSpec,
    OpenMLDatasetSpec,
    SyntheticGaussianKernelDatasetSpec,
    SyntheticSpectrumDatasetSpec,
)
from .kernels import GaussianKernel, KernelFunction
from .operators import DensePSDOperator, KernelPSDOperator, PSDOperator
from .results import ApproximationResult

__all__ = [
    "ApproximationResult",
    "CURATED_OPENML_DATASETS",
    "DatasetBundle",
    "DatasetSpec",
    "DensePSDOperator",
    "GaussianKernel",
    "KernelFunction",
    "KernelPSDOperator",
    "OpenMLDatasetSpec",
    "PSDOperator",
    "SyntheticGaussianKernelDatasetSpec",
    "SyntheticSpectrumDatasetSpec",
]

__version__ = "0.1.0"
