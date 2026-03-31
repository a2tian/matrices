# Extending matrices

## Add a method

Implement `matrices.methods.ApproximationMethod` and register it in `matrices.methods.BUILTIN_METHODS`.

Method implementations should:

- accept any `PSDOperator`
- use the provided `numpy.random.Generator`
- return an `ApproximationResult`
- avoid benchmark-specific logic

## Add a dataset

Implement `matrices.datasets.DatasetSpec` and return a `DatasetBundle`.

Dataset specs should encapsulate sampling, standardization, and operator construction so benchmark configs stay declarative.

## Add a benchmark suite

Create a new TOML file under `src/matrices/benchmarks/suites` or pass a standalone config path to `matrices-bench run`.
