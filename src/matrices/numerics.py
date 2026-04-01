from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def symmetrize(matrix: FloatArray) -> FloatArray:
    return 0.5 * (matrix + matrix.T)


def orthonormal_column_basis(columns: FloatArray, *, rcond: float = 1e-10) -> FloatArray:
    """Return an orthonormal basis for the column span of columns."""

    matrix = np.asarray(columns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("columns must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)

    left_vectors, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    max_singular_value = float(np.max(singular_values, initial=0.0))
    threshold = rcond * max(max_singular_value, 1.0)
    keep = singular_values > threshold
    return np.asarray(left_vectors[:, keep], dtype=float)


def nystrom_factor(
    columns: FloatArray,
    intersection: FloatArray,
    *,
    rcond: float = 1e-10,
) -> FloatArray:
    """Return a factor F such that F F^T is the Nyström approximation."""

    basis = np.asarray(columns, dtype=float)
    if basis.ndim != 2:
        raise ValueError("columns must be two-dimensional")

    block = symmetrize(np.asarray(intersection, dtype=float))
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError("intersection must be square")
    if block.size == 0:
        return np.zeros((basis.shape[0], 0), dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(block)
    max_eigenvalue = float(np.max(np.abs(eigenvalues), initial=0.0))
    threshold = rcond * max(max_eigenvalue, 1.0)
    keep = eigenvalues > threshold
    if not np.any(keep):
        return np.zeros((basis.shape[0], 0), dtype=float)

    inv_sqrt = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
    return np.asarray(basis @ inv_sqrt, dtype=float)


def psd_eigendecomposition(matrix: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return eigenvalues and eigenvectors sorted in descending order."""

    eigenvalues, eigenvectors = np.linalg.eigh(symmetrize(np.asarray(matrix, dtype=float)))
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def optimal_psd_approximation(
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    rank: int,
) -> FloatArray:
    """Construct the best rank-k PSD approximation from an eigendecomposition."""

    truncated_rank = min(rank, eigenvalues.size)
    if truncated_rank <= 0:
        return np.zeros((eigenvectors.shape[0], eigenvectors.shape[0]), dtype=float)
    values = np.clip(eigenvalues[:truncated_rank], a_min=0.0, a_max=None)
    vectors = eigenvectors[:, :truncated_rank]
    return (vectors * values) @ vectors.T
