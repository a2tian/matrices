from __future__ import annotations

import csv
from pathlib import Path

from matrices.benchmarks import BenchmarkConfig, BenchmarkRunner, available_suites


def test_available_suites_expose_bundled_configs() -> None:
    assert {suite.name for suite in available_suites()} == {"openml_full", "reference", "smoke"}


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

    runner.report(artifacts.output_dir)

    with artifacts.summary_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
