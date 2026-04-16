from __future__ import annotations

import os

import numpy as np
import pytest

from matrices.datasets import (
    OpenMLDatasetSpec,
    SyntheticGaussianKernelDatasetSpec,
    SyntheticIdentityPlusOnesDatasetSpec,
    SyntheticSpectrumDatasetSpec,
)


def test_synthetic_dataset_specs_build_operators() -> None:
    dense_bundle = SyntheticSpectrumDatasetSpec(name="spectrum").build()
    identity_plus_ones_bundle = SyntheticIdentityPlusOnesDatasetSpec(
        name="identity_plus_ones",
        size=4,
        delta=0.25,
    ).build()
    kernel_bundle = SyntheticGaussianKernelDatasetSpec(name="gaussian").build()

    assert dense_bundle.operator.shape == (64, 64)
    assert identity_plus_ones_bundle.operator.shape == (4, 4)
    assert kernel_bundle.operator.shape == (64, 64)
    np.testing.assert_allclose(
        identity_plus_ones_bundle.operator.materialize(),
        np.array(
            [
                [1.25, 0.25, 0.25, 0.25],
                [0.25, 1.25, 0.25, 0.25],
                [0.25, 0.25, 1.25, 0.25],
                [0.25, 0.25, 0.25, 1.25],
            ],
            dtype=float,
        ),
    )


def test_identity_plus_ones_dataset_spec_rejects_non_psd_delta() -> None:
    with pytest.raises(ValueError, match="delta must be at least"):
        SyntheticIdentityPlusOnesDatasetSpec(
            name="invalid",
            size=4,
            delta=-0.3,
        ).build()


@pytest.mark.openml
def test_openml_dataset_spec_builds_operator() -> None:
    if os.environ.get("RUN_OPENML") != "1":
        pytest.skip("set RUN_OPENML=1 to enable OpenML integration tests")

    bundle = OpenMLDatasetSpec(name="mnist", dataset="mnist_784", n_samples=32).build()
    assert bundle.operator.shape == (32, 32)
