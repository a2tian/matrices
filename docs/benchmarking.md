# Benchmarking

The `matrices-bench` CLI loads benchmark suites from TOML files.

## Built-in suites

- `smoke`: fast synthetic checks intended for local development and CI
- `reference`: small synthetic and OpenML workloads with exact PSD reference curves
- `openml_full`: larger OpenML workloads without exact-reference baselines

## CLI

```bash
matrices-bench list
matrices-bench run smoke
matrices-bench run path/to/config.toml --output-dir results/custom
matrices-bench report results/custom
```

## Dataset Kinds

Benchmark configs currently support `synthetic_spectrum`, `synthetic_identity_plus_ones`, `synthetic_gaussian_kernel`, and `openml`.

Example:

```toml
[[datasets]]
kind = "synthetic_identity_plus_ones"
name = "identity_plus_ones_small"
size = 128
delta = 0.05
```

## Outputs

Each run writes:

- `trial_results.csv`
- `summary.csv`
- `summary.json`
- `report.md`
- plot PNGs
- the resolved benchmark config used for the run
