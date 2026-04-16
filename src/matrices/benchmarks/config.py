from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datasets import (
    DatasetSpec,
    OpenMLDatasetSpec,
    SyntheticGaussianKernelDatasetSpec,
    SyntheticIdentityPlusOnesDatasetSpec,
    SyntheticSpectrumDatasetSpec,
)


@dataclass(slots=True, frozen=True)
class SuiteInfo:
    name: str
    description: str


SUITES_DIR = Path(__file__).resolve().parent / "suites"


def _require_list(values: Any, field_name: str) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a non-empty list")
    return values


def _load_config_text(path_or_suite: str | Path) -> str:
    candidate = Path(path_or_suite)
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    suite_name = str(path_or_suite).removesuffix(".toml")
    suite_path = SUITES_DIR / f"{suite_name}.toml"
    if not suite_path.is_file():
        raise FileNotFoundError(f"could not find benchmark config '{path_or_suite}'")
    return suite_path.read_text(encoding="utf-8")


def available_suites() -> tuple[SuiteInfo, ...]:
    suites: list[SuiteInfo] = []
    for suite_path in sorted(SUITES_DIR.iterdir(), key=lambda item: item.name):
        if suite_path.suffix != ".toml":
            continue
        data = tomllib.loads(suite_path.read_text(encoding="utf-8"))
        suites.append(
            SuiteInfo(
                name=str(data["name"]),
                description=str(data.get("description", "")),
            )
        )
    return tuple(suites)


def _build_dataset_spec(raw: dict[str, Any]) -> DatasetSpec:
    dataset_type = str(raw["kind"])
    if dataset_type == "synthetic_spectrum":
        return SyntheticSpectrumDatasetSpec(
            name=str(raw["name"]),
            size=int(raw.get("size", 64)),
            decay=str(raw.get("decay", "polynomial")),
            alpha=float(raw.get("alpha", 1.5)),
            seed=int(raw.get("seed", 0)),
            jitter=float(raw.get("jitter", 1e-8)),
        )
    if dataset_type == "synthetic_identity_plus_ones":
        return SyntheticIdentityPlusOnesDatasetSpec(
            name=str(raw["name"]),
            size=int(raw.get("size", 64)),
            delta=float(raw.get("delta", 0.1)),
        )
    if dataset_type == "synthetic_gaussian_kernel":
        bandwidth_value = raw.get("bandwidth")
        bandwidth = float(bandwidth_value) if bandwidth_value is not None else None
        return SyntheticGaussianKernelDatasetSpec(
            name=str(raw["name"]),
            n_samples=int(raw.get("n_samples", 64)),
            n_features=int(raw.get("n_features", 8)),
            clusters=int(raw.get("clusters", 3)),
            cluster_std=float(raw.get("cluster_std", 0.7)),
            bandwidth=bandwidth,
            seed=int(raw.get("seed", 0)),
        )
    if dataset_type == "openml":
        bandwidth_value = raw.get("bandwidth")
        bandwidth = float(bandwidth_value) if bandwidth_value is not None else None
        return OpenMLDatasetSpec(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            n_samples=int(raw.get("n_samples", 512)),
            seed=int(raw.get("seed", 0)),
            bandwidth=bandwidth,
            standardize=bool(raw.get("standardize", True)),
            cache_dir=str(raw.get("cache_dir", ".cache/openml")),
        )
    raise ValueError(f"unsupported dataset kind '{dataset_type}'")


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value: {value!r}")


@dataclass(slots=True, frozen=True)
class BenchmarkConfig:
    name: str
    description: str
    methods: tuple[str, ...]
    ranks: tuple[int, ...]
    seeds: tuple[int, ...]
    datasets: tuple[DatasetSpec, ...]
    output_dir: str | None = None
    reference_enabled: bool = False

    @classmethod
    def from_toml(cls, path_or_suite: str | Path) -> BenchmarkConfig:
        data = tomllib.loads(_load_config_text(path_or_suite))
        dataset_rows = _require_list(data.get("datasets"), "datasets")
        datasets = tuple(_build_dataset_spec(dict(item)) for item in dataset_rows)
        reference = data.get("reference", {})
        if reference is None:
            reference = {}
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            methods=tuple(str(item) for item in _require_list(data.get("methods"), "methods")),
            ranks=tuple(int(item) for item in _require_list(data.get("ranks"), "ranks")),
            seeds=tuple(int(item) for item in data.get("seeds", list(range(10)))),
            datasets=datasets,
            output_dir=str(data["output_dir"]) if "output_dir" in data else None,
            reference_enabled=bool(reference.get("enabled", False)),
        )

    def to_toml(self) -> str:
        lines = [
            f"name = {_toml_value(self.name)}",
            f"description = {_toml_value(self.description)}",
            f"methods = {_toml_value(list(self.methods))}",
            f"ranks = {_toml_value(list(self.ranks))}",
            f"seeds = {_toml_value(list(self.seeds))}",
        ]
        if self.output_dir is not None:
            lines.append(f"output_dir = {_toml_value(self.output_dir)}")
        lines.extend(["", "[reference]", f"enabled = {_toml_value(self.reference_enabled)}"])
        for dataset in self.datasets:
            lines.extend(["", "[[datasets]]"])
            for key, value in dataset.to_config().items():
                lines.append(f"{key} = {_toml_value(value)}")
        return "\n".join(lines) + "\n"
