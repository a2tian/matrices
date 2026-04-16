from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .kernels import GaussianKernel
from .operators import DensePSDOperator, KernelPSDOperator, PSDOperator

FloatArray = NDArray[np.float64]

CURATED_OPENML_DATASETS: dict[str, str] = {
    "creditcard": "creditcard",
    "hls4ml_lhc_jets_hlf": "hls4ml_lhc_jets_hlf",
    "jannis": "jannis",
    "mnist_784": "mnist_784",
    "volkert": "volkert",
    "yolanda": "Yolanda",
}


@dataclass(slots=True)
class DatasetBundle:
    name: str
    kind: str
    operator: PSDOperator
    metadata: dict[str, object] = field(default_factory=dict)


class DatasetSpec(ABC):
    """Declarative dataset builder used by the benchmark runner."""

    name: str
    kind: str

    @abstractmethod
    def build(self) -> DatasetBundle:
        """Create a dataset bundle with a PSD operator."""

    @abstractmethod
    def to_config(self) -> dict[str, object]:
        """Serialize the spec into a TOML-compatible dictionary."""


def _standardize_features(features: FloatArray) -> FloatArray:
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    standardized = np.divide(
        features - mean,
        std,
        out=np.zeros_like(features, dtype=float),
        where=std != 0,
    )
    return np.asarray(standardized, dtype=float)


@dataclass(slots=True)
class SyntheticSpectrumDatasetSpec(DatasetSpec):
    name: str
    size: int = 64
    decay: str = "polynomial"
    alpha: float = 1.5
    seed: int = 0
    jitter: float = 1e-8
    kind: str = "synthetic_spectrum"

    def _eigenvalues(self) -> FloatArray:
        indices = np.arange(1, self.size + 1, dtype=float)
        if self.decay == "polynomial":
            return indices ** (-self.alpha)
        if self.decay == "exponential":
            return np.exp(-self.alpha * (indices - 1.0) / self.size)
        raise ValueError(f"unsupported decay: {self.decay}")

    def build(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        basis, _ = np.linalg.qr(rng.standard_normal((self.size, self.size)))
        eigenvalues = self._eigenvalues()
        matrix = (basis * eigenvalues) @ basis.T
        matrix = matrix + self.jitter * np.eye(self.size)
        return DatasetBundle(
            name=self.name,
            kind=self.kind,
            operator=DensePSDOperator(matrix),
            metadata={
                "alpha": self.alpha,
                "decay": self.decay,
                "jitter": self.jitter,
                "size": self.size,
            },
        )

    def to_config(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "size": self.size,
            "decay": self.decay,
            "alpha": self.alpha,
            "seed": self.seed,
            "jitter": self.jitter,
        }


@dataclass(slots=True)
class SyntheticIdentityPlusOnesDatasetSpec(DatasetSpec):
    name: str
    size: int = 64
    delta: float = 0.1
    kind: str = "synthetic_identity_plus_ones"

    def build(self) -> DatasetBundle:
        if self.size <= 0:
            raise ValueError("size must be positive")
        min_delta = -1.0 / self.size
        if self.delta < min_delta:
            raise ValueError(
                f"delta must be at least {min_delta} for I + delta J to remain PSD"
            )
        matrix = np.eye(self.size, dtype=float) + self.delta * np.ones(
            (self.size, self.size),
            dtype=float,
        )
        return DatasetBundle(
            name=self.name,
            kind=self.kind,
            operator=DensePSDOperator(matrix),
            metadata={
                "delta": self.delta,
                "size": self.size,
            },
        )

    def to_config(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "size": self.size,
            "delta": self.delta,
        }


@dataclass(slots=True)
class SyntheticGaussianKernelDatasetSpec(DatasetSpec):
    name: str
    n_samples: int = 64
    n_features: int = 8
    clusters: int = 3
    cluster_std: float = 0.7
    bandwidth: float | None = None
    seed: int = 0
    kind: str = "synthetic_gaussian_kernel"

    def build(self) -> DatasetBundle:
        rng = np.random.default_rng(self.seed)
        centers = rng.normal(scale=3.0, size=(self.clusters, self.n_features))
        assignments = rng.integers(0, self.clusters, size=self.n_samples)
        noise = rng.normal(scale=self.cluster_std, size=(self.n_samples, self.n_features))
        points = centers[assignments] + noise
        bandwidth = (
            self.bandwidth if self.bandwidth is not None else float(np.sqrt(self.n_features))
        )
        return DatasetBundle(
            name=self.name,
            kind=self.kind,
            operator=KernelPSDOperator(points, GaussianKernel(bandwidth=bandwidth)),
            metadata={
                "bandwidth": bandwidth,
                "clusters": self.clusters,
                "n_features": self.n_features,
                "n_samples": self.n_samples,
            },
        )

    def to_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "kind": self.kind,
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "clusters": self.clusters,
            "cluster_std": self.cluster_std,
            "seed": self.seed,
        }
        if self.bandwidth is not None:
            config["bandwidth"] = self.bandwidth
        return config


@dataclass(slots=True)
class OpenMLDatasetSpec(DatasetSpec):
    name: str
    dataset: str
    n_samples: int = 512
    seed: int = 0
    bandwidth: float | None = None
    standardize: bool = True
    cache_dir: str = ".cache/openml"
    kind: str = "openml"

    def build(self) -> DatasetBundle:
        if self.dataset not in CURATED_OPENML_DATASETS:
            raise ValueError(
                f"dataset '{self.dataset}' is not in the curated OpenML registry: "
                f"{sorted(CURATED_OPENML_DATASETS)}"
            )

        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:
            raise RuntimeError("OpenML support requires `pip install -e .[bench]`") from exc

        features, _ = fetch_openml(
            name=CURATED_OPENML_DATASETS[self.dataset],
            return_X_y=True,
            as_frame=False,
            parser="auto",
            data_home=str(Path(self.cache_dir)),
        )
        features = np.asarray(features, dtype=float)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(features.shape[0])[: self.n_samples]
        sampled = features[indices]
        if self.standardize:
            sampled = _standardize_features(sampled)
        bandwidth = (
            self.bandwidth if self.bandwidth is not None else float(np.sqrt(sampled.shape[1]))
        )
        return DatasetBundle(
            name=self.name,
            kind=self.kind,
            operator=KernelPSDOperator(sampled, GaussianKernel(bandwidth=bandwidth)),
            metadata={
                "bandwidth": bandwidth,
                "dataset": self.dataset,
                "n_features": sampled.shape[1],
                "n_samples": sampled.shape[0],
                "standardize": self.standardize,
            },
        )

    def to_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "kind": self.kind,
            "name": self.name,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "standardize": self.standardize,
            "cache_dir": self.cache_dir,
        }
        if self.bandwidth is not None:
            config["bandwidth"] = self.bandwidth
        return config
