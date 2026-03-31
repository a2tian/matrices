from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..methods import build_method
from ..metrics import evaluate_approximation
from ..numerics import optimal_psd_approximation, psd_eigendecomposition
from ..operators import CountingPSDOperator
from .config import BenchmarkConfig
from .reporting import (
    generate_plots,
    summarize_trials,
    write_markdown_report,
    write_summary,
)


def _require_benchmark_dependencies() -> Any:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Benchmarking requires `pip install -e .[bench]`") from exc
    return pd


@dataclass(slots=True, frozen=True)
class BenchmarkArtifacts:
    output_dir: Path
    trial_results_path: Path
    summary_csv_path: Path
    summary_json_path: Path
    report_path: Path
    plot_paths: tuple[Path, ...]
    reference_results_path: Path | None = None


class BenchmarkRunner:
    """Execute benchmark suites and produce reports."""

    def run(self, config: BenchmarkConfig, *, output_dir: Path | None = None) -> BenchmarkArtifacts:
        pd = _require_benchmark_dependencies()
        resolved_output = self._resolve_output_dir(config, output_dir)
        resolved_output.mkdir(parents=True, exist_ok=True)
        (resolved_output / "config.toml").write_text(config.to_toml(), encoding="utf-8")

        trial_rows: list[dict[str, object]] = []
        reference_rows: list[dict[str, object]] = []

        for dataset in config.datasets:
            bundle = dataset.build()
            target = bundle.operator.materialize()
            reference_by_rank: dict[int, tuple[float, float]] = {}
            if config.reference_enabled:
                eigenvalues, eigenvectors = psd_eigendecomposition(target)
                for rank in config.ranks:
                    reference_matrix = optimal_psd_approximation(eigenvalues, eigenvectors, rank)
                    reference_metrics = evaluate_approximation(target, reference_matrix)
                    reference_by_rank[rank] = (
                        reference_metrics.relative_frobenius_error,
                        reference_metrics.relative_trace_error,
                    )
                    reference_rows.append(
                        {
                            "dataset": bundle.name,
                            "rank": rank,
                            "relative_frobenius_error": reference_metrics.relative_frobenius_error,
                            "relative_trace_error": reference_metrics.relative_trace_error,
                        }
                    )

            for method_name in config.methods:
                method = build_method(method_name)
                seeds = (0,) if method.deterministic else config.seeds
                for rank in config.ranks:
                    for seed in seeds:
                        counted_operator = CountingPSDOperator(bundle.operator)
                        result = method.run(
                            counted_operator,
                            rank=rank,
                            rng=np.random.default_rng(seed),
                        )
                        approximation = result.materialize()
                        metrics = evaluate_approximation(target, approximation)
                        reference_values = reference_by_rank.get(rank)
                        row = {
                            "dataset": bundle.name,
                            "dataset_kind": bundle.kind,
                            "operator_type": type(bundle.operator).__name__,
                            "metadata_json": json.dumps(bundle.metadata, sort_keys=True),
                            "method": result.method,
                            "rank": rank,
                            "seed": seed,
                            "runtime_seconds": result.runtime_seconds,
                            "effective_rank": result.effective_rank,
                            "entry_evaluations": counted_operator.entry_evaluations,
                            "selected_count": len(result.selected_indices),
                            "selected_indices": "|".join(
                                str(index) for index in result.selected_indices
                            ),
                            "relative_frobenius_error": metrics.relative_frobenius_error,
                            "relative_trace_error": metrics.relative_trace_error,
                            "reference_rel_frobenius_error": (
                                reference_values[0] if reference_values is not None else np.nan
                            ),
                            "reference_rel_trace_error": (
                                reference_values[1] if reference_values is not None else np.nan
                            ),
                        }
                        trial_rows.append(row)

        trials_df = pd.DataFrame(trial_rows)
        trial_results_path = resolved_output / "trial_results.csv"
        trials_df.to_csv(trial_results_path, index=False)

        reference_results_path: Path | None = None
        if reference_rows:
            reference_results_path = resolved_output / "reference_results.csv"
            pd.DataFrame(reference_rows).to_csv(reference_results_path, index=False)

        summary_df = summarize_trials(trials_df)
        summary_csv_path, summary_json_path = write_summary(summary_df, resolved_output)
        plot_paths = generate_plots(summary_df, resolved_output)
        report_path = write_markdown_report(config, summary_df, resolved_output)

        return BenchmarkArtifacts(
            output_dir=resolved_output,
            trial_results_path=trial_results_path,
            summary_csv_path=summary_csv_path,
            summary_json_path=summary_json_path,
            report_path=report_path,
            plot_paths=plot_paths,
            reference_results_path=reference_results_path,
        )

    def report(self, results_dir: Path) -> BenchmarkArtifacts:
        pd = _require_benchmark_dependencies()
        config = BenchmarkConfig.from_toml(results_dir / "config.toml")
        trials_df = pd.read_csv(results_dir / "trial_results.csv")
        summary_df = summarize_trials(trials_df)
        summary_csv_path, summary_json_path = write_summary(summary_df, results_dir)
        plot_paths = generate_plots(summary_df, results_dir)
        report_path = write_markdown_report(config, summary_df, results_dir)
        reference_path = results_dir / "reference_results.csv"
        return BenchmarkArtifacts(
            output_dir=results_dir,
            trial_results_path=results_dir / "trial_results.csv",
            summary_csv_path=summary_csv_path,
            summary_json_path=summary_json_path,
            report_path=report_path,
            plot_paths=plot_paths,
            reference_results_path=reference_path if reference_path.exists() else None,
        )

    def _resolve_output_dir(self, config: BenchmarkConfig, output_dir: Path | None) -> Path:
        if output_dir is not None:
            return output_dir
        if config.output_dir is not None:
            return Path(config.output_dir)
        return Path("results") / config.name
