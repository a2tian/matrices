from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .benchmarks import BenchmarkConfig, BenchmarkRunner, available_suites


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="matrices-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List bundled benchmark suites")

    run_parser = subparsers.add_parser("run", help="Run a benchmark suite or TOML config")
    run_parser.add_argument("config", help="Built-in suite name or path to a TOML config")
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate reports from an existing results directory",
    )
    report_parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing trial_results.csv and config.toml",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for suite in available_suites():
            print(f"{suite.name}: {suite.description}")
        return 0

    runner = BenchmarkRunner()
    if args.command == "run":
        config = BenchmarkConfig.from_toml(args.config)
        artifacts = runner.run(config, output_dir=args.output_dir)
        print(f"wrote benchmark results to {artifacts.output_dir}")
        return 0

    artifacts = runner.report(args.results_dir)
    print(f"updated reports in {artifacts.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
