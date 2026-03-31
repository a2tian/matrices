from __future__ import annotations

import os

import pytest

from matrices.datasets import (
    OpenMLDatasetSpec,
    SyntheticGaussianKernelDatasetSpec,
    SyntheticSpectrumDatasetSpec,
)


def test_synthetic_dataset_specs_build_operators() -> None:
    dense_bundle = SyntheticSpectrumDatasetSpec(name="spectrum").build()
    kernel_bundle = SyntheticGaussianKernelDatasetSpec(name="gaussian").build()

    assert dense_bundle.operator.shape == (64, 64)
    assert kernel_bundle.operator.shape == (64, 64)


@pytest.mark.openml
def test_openml_dataset_spec_builds_operator() -> None:
    if os.environ.get("RUN_OPENML") != "1":
        pytest.skip("set RUN_OPENML=1 to enable OpenML integration tests")

    bundle = OpenMLDatasetSpec(name="mnist", dataset="mnist_784", n_samples=32).build()
    assert bundle.operator.shape == (32, 32)
