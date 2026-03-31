# matrices

`matrices` provides reusable implementations of adaptive sampling methods for low-rank approximation of positive semidefinite matrices.

## Design goals

- Clean abstractions for dense and lazy PSD operators
- Reproducible benchmark orchestration through TOML configs
- Extensible method and dataset interfaces
- CPU-first implementations with clear numerical behavior

## Install

```bash
python3 -m pip install -e .[dev]
```

## Library example

```python
import numpy as np

from matrices.methods import UniformNystromMethod
from matrices.operators import DensePSDOperator

rng = np.random.default_rng(42)
x = rng.standard_normal((40, 10))
operator = DensePSDOperator(x @ x.T)

result = UniformNystromMethod().run(operator, rank=8, rng=rng)
print(result.effective_rank)
```
