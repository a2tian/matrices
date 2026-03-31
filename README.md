# matrices

`matrices` is a Python package for implementing and benchmarking adaptive sampling methods for low-rank approximation of positive semidefinite matrices.

The library is built around a generic PSD operator interface, so the same methods work on explicit dense matrices and lazy kernel operators. The benchmark tooling is config-driven and reproducible, with built-in synthetic and OpenML-backed suites.

## Installation

```bash
python3 -m pip install -e .[dev]
```

To install the benchmark extras only:

```bash
python3 -m pip install -e .[bench]
```

## Quickstart

```python
import numpy as np

from matrices.methods import GreedyCholeskyMethod
from matrices.operators import DensePSDOperator

rng = np.random.default_rng(0)
a = rng.standard_normal((32, 8))
matrix = a @ a.T
operator = DensePSDOperator(matrix)

method = GreedyCholeskyMethod()
result = method.run(operator, rank=6, rng=np.random.default_rng(0))
approximation = result.materialize()
```

## Benchmark CLI

List the built-in suites:

```bash
matrices-bench list
```

Run a built-in suite or a custom TOML config:

```bash
matrices-bench run smoke
matrices-bench run path/to/config.toml --output-dir results/smoke
```

Regenerate reports from an existing results directory:

```bash
matrices-bench report results/smoke
```

## Documentation

- User guide: `mkdocs serve`
- Benchmark suites: see `src/matrices/benchmarks/suites`
- Extension points: see `docs/extending.md`

## References

[1] Yifan Chen, Ethan N. Epperly, Joel A. Tropp, and Robert J. Webber. Randomly pivoted Cholesky: Practical approximation of a kernel matrix with few entry evaluations. Communications on Pure and Applied Mathematics, 78(5):995-1041, 2025.
