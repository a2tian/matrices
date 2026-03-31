from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig


def _require_reporting_dependencies() -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("Reporting requires `pip install -e .[bench]`") from exc
    return pd, plt


def summarize_trials(trials_df: Any) -> Any:
    summary = trials_df.groupby(["dataset", "method", "rank"], as_index=False).agg(
        dataset_kind=("dataset_kind", "first"),
        operator_type=("operator_type", "first"),
        trials=("seed", "count"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        runtime_seconds_std=("runtime_seconds", "std"),
        relative_frobenius_error_mean=("relative_frobenius_error", "mean"),
        relative_frobenius_error_std=("relative_frobenius_error", "std"),
        relative_trace_error_mean=("relative_trace_error", "mean"),
        relative_trace_error_std=("relative_trace_error", "std"),
        entry_evaluations_mean=("entry_evaluations", "mean"),
        effective_rank_mean=("effective_rank", "mean"),
        reference_rel_frobenius_error=("reference_rel_frobenius_error", "first"),
        reference_rel_trace_error=("reference_rel_trace_error", "first"),
    )
    for column_name in [
        "runtime_seconds_std",
        "relative_frobenius_error_std",
        "relative_trace_error_std",
    ]:
        summary[column_name] = summary[column_name].fillna(0.0)
    return summary


def write_summary(summary_df: Any, output_dir: Path) -> tuple[Path, Path]:
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(json.loads(summary_df.to_json(orient="records")), indent=2),
        encoding="utf-8",
    )
    return summary_csv, summary_json


def _format_cell(value: object) -> str:
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None and pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _dataframe_to_markdown(dataframe: Any) -> str:
    headers = list(dataframe.columns)
    rows = [
        [_format_cell(item) for item in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_plots(summary_df: Any, output_dir: Path) -> tuple[Path, ...]:
    _, plt = _require_reporting_dependencies()
    plot_paths: list[Path] = []
    metrics = [
        ("relative_frobenius_error_mean", "Relative Frobenius Error", "frobenius"),
        ("relative_trace_error_mean", "Relative Trace Error", "trace"),
        ("runtime_seconds_mean", "Runtime (s)", "runtime"),
    ]
    for dataset_name, dataset_frame in summary_df.groupby("dataset"):
        for metric_column, title, suffix in metrics:
            figure, axis = plt.subplots(figsize=(8, 5))
            for method_name, method_frame in dataset_frame.groupby("method"):
                axis.plot(
                    method_frame["rank"],
                    method_frame[metric_column],
                    marker="o",
                    label=method_name,
                )
            axis.set_title(f"{dataset_name}: {title}")
            axis.set_xlabel("Rank")
            axis.set_ylabel(title)
            axis.legend()
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            plot_path = output_dir / f"{dataset_name}_{suffix}.png"
            figure.savefig(plot_path, dpi=160)
            plt.close(figure)
            plot_paths.append(plot_path)
    return tuple(plot_paths)


def write_markdown_report(
    config: BenchmarkConfig,
    summary_df: Any,
    output_dir: Path,
) -> Path:
    report_path = output_dir / "report.md"
    max_rank = int(summary_df["rank"].max())
    best_frobenius = summary_df[summary_df["rank"] == max_rank].sort_values(
        ["dataset", "relative_frobenius_error_mean", "runtime_seconds_mean"]
    )
    best_frobenius = best_frobenius.groupby("dataset", as_index=False).first()[
        ["dataset", "method", "relative_frobenius_error_mean", "runtime_seconds_mean"]
    ]
    best_trace = summary_df[summary_df["rank"] == max_rank].sort_values(
        ["dataset", "relative_trace_error_mean", "runtime_seconds_mean"]
    )
    best_trace = best_trace.groupby("dataset", as_index=False).first()[
        ["dataset", "method", "relative_trace_error_mean", "runtime_seconds_mean"]
    ]
    lines = [
        "# Benchmark Report",
        "",
        f"- Suite: `{config.name}`",
        f"- Description: {config.description}",
        f"- Methods: {', '.join(config.methods)}",
        f"- Ranks: {', '.join(str(rank) for rank in config.ranks)}",
        f"- Datasets: {', '.join(dataset.name for dataset in config.datasets)}",
        f"- Reference enabled: {config.reference_enabled}",
        "",
        "## Best Frobenius Error at Max Rank",
        "",
        _dataframe_to_markdown(best_frobenius),
        "",
        "## Best Trace Error at Max Rank",
        "",
        _dataframe_to_markdown(best_trace),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
