from __future__ import annotations

import csv
from pathlib import Path

from matrices.benchmarks import BenchmarkConfig, BenchmarkRunner, available_suites


def test_available_suites_expose_bundled_configs() -> None:
    suite_names = {suite.name for suite in available_suites()}

    assert {"openml_full", "reference", "smoke"} <= suite_names


def test_smoke_suite_run_and_report(tmp_path: Path) -> None:
    config = BenchmarkConfig.from_toml("smoke")
    runner = BenchmarkRunner()

    artifacts = runner.run(config, output_dir=tmp_path / "smoke")
    assert artifacts.trial_results_path.exists()
    assert artifacts.summary_csv_path.exists()
    assert artifacts.summary_json_path.exists()
    assert artifacts.report_path.exists()
    assert artifacts.reference_results_path is not None
    assert artifacts.reference_results_path.exists()
    assert artifacts.plot_paths
    assert any(path.name.endswith("_entries.png") for path in artifacts.plot_paths)

    runner.report(artifacts.output_dir)

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows


def test_config_parses_identity_plus_ones_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "identity_plus_ones.toml"
    config_path.write_text(
        "\n".join(
            [
                'name = "identity_plus_ones"',
                'description = "Synthetic identity plus ones family"',
                'methods = ["uniform_nystrom"]',
                "ranks = [4]",
                "",
                "[reference]",
                "enabled = false",
                "",
                "[[datasets]]",
                'kind = "synthetic_identity_plus_ones"',
                'name = "identity_plus_ones_small"',
                "size = 8",
                "delta = 0.2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = BenchmarkConfig.from_toml(config_path)

    assert len(config.datasets) == 1
    dataset = config.datasets[0]
    assert dataset.kind == "synthetic_identity_plus_ones"
    assert dataset.to_config()["delta"] == 0.2
