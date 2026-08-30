"""Reusable expression-binning utilities used by K-SPADA tutorials."""

from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
from scipy import sparse


def to_dense_float32(matrix_object) -> np.ndarray:
    """Convert a sparse or dense expression matrix to a dense float32 array."""
    if sparse.issparse(matrix_object):
        return matrix_object.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix_object, dtype=np.float32)


def digitize_like_scgpt(
    values: np.ndarray,
    bin_edges: np.ndarray,
    random_generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Reproduce scGPT tie handling between left and right digitization results."""
    if values.ndim != 1 or bin_edges.ndim != 1:
        raise ValueError("scGPT digitization expects one-dimensional values and bin edges")
    left_digits = np.digitize(values, bin_edges)
    right_digits = np.digitize(values, bin_edges, right=True)
    generator = random_generator or np.random.default_rng()
    random_values = generator.random(len(values))
    interpolated_digits = random_values * (right_digits - left_digits) + left_digits
    return np.ceil(interpolated_digits).astype(np.int64)


def bin_expression_row(
    expression_row: np.ndarray,
    bin_count: int = 8,
    random_generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Preserve zero values and quantile-bin nonzero expression into ``1..bin_count-1``."""
    if bin_count < 2:
        raise ValueError("bin_count must be at least 2")
    row_values = np.asarray(expression_row)
    if row_values.ndim != 1:
        raise ValueError("expression_row must be one-dimensional")
    if row_values.size == 0 or row_values.max() == 0:
        return np.zeros_like(row_values, dtype=np.int64)

    binned_row = np.zeros_like(row_values, dtype=np.int64)
    nonzero_indices = np.flatnonzero(row_values)
    values_to_bin = row_values[nonzero_indices] if row_values.min() <= 0 else row_values
    bin_edges = np.quantile(values_to_bin, np.linspace(0, 1, bin_count - 1))
    binned_values = digitize_like_scgpt(values_to_bin, bin_edges, random_generator=random_generator)
    if row_values.min() <= 0:
        binned_row[nonzero_indices] = binned_values
        return binned_row
    return binned_values


def bin_expression_matrix(
    expression_matrix,
    bin_count: int = 8,
    random_seed: int = 1,
) -> np.ndarray:
    """Apply reproducible scGPT-style binning independently to every cell."""
    dense_matrix = to_dense_float32(expression_matrix)
    random_generator = np.random.default_rng(random_seed)
    binned_matrix = np.empty(dense_matrix.shape, dtype=np.int64)
    for cell_index in range(dense_matrix.shape[0]):
        binned_matrix[cell_index] = bin_expression_row(
            dense_matrix[cell_index],
            bin_count=bin_count,
            random_generator=random_generator,
        )
    return binned_matrix


def bin_adata(
    adata_input: ad.AnnData,
    bin_count: int = 8,
    random_seed: int = 1,
    copy: bool = True,
) -> ad.AnnData:
    """Return AnnData whose expression matrix contains reproducible integer bins."""
    binned_adata = adata_input.copy() if copy else adata_input
    binned_adata.X = bin_expression_matrix(
        binned_adata.X,
        bin_count=bin_count,
        random_seed=random_seed,
    )
    binned_adata.uns["kspada_binning"] = {
        "method": "scgpt_quantile_binning_with_zero_preserved",
        "bin_count": int(bin_count),
        "random_seed": int(random_seed),
        "value_range": [0, int(bin_count - 1)],
    }
    return binned_adata
