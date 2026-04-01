# Extending matrices

## Add a method

Implement `matrices.methods.ApproximationMethod` and register it in `matrices.methods.BUILTIN_METHODS`.

Method implementations should:

- accept any `PSDOperator`
- rely on the public operator surface (`diagonal()`, `entry()`, `column()`, `submatrix()`, `materialize()`) instead of assuming a dense matrix
- use the provided `numpy.random.Generator`
- return an `ApproximationResult`
- avoid benchmark-specific logic

Built-in Cholesky pivoting uses lazy operator queries while selecting pivots. Selector logic should not materialize the full matrix or residual.

## Add a dataset

Implement `matrices.datasets.DatasetSpec` and return a `DatasetBundle`.

Dataset specs should encapsulate sampling, standardization, and operator construction so benchmark configs stay declarative.

## Add a benchmark suite

Create a new TOML file under `src/matrices/benchmarks/suites` or pass a standalone config path to `matrices-bench run`.
