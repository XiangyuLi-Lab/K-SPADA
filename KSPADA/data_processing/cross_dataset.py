"""Prepare two generic datasets for cross-dataset K-SPADA prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union

import anndata as ad

from . import align_adatas_to_union_genes, prepare_data_workflow, standardize_obs_columns


PathLike = Union[str, Path]
DEFAULT_DATA_DIRECTORY = Path("/your_path/example_dataset")
DEFAULT_REFERENCE_PATH = DEFAULT_DATA_DIRECTORY / "reference_source.h5ad"
DEFAULT_QUERY_PATH = DEFAULT_DATA_DIRECTORY / "query_source.h5ad"
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_DATA_DIRECTORY / "cross_dataset_prepared"


def load_and_align_datasets(
    reference_path: PathLike = DEFAULT_REFERENCE_PATH,
    query_path: PathLike = DEFAULT_QUERY_PATH,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Load two AnnData files and align them to one ordered gene space.

    Args:
        reference_path: AnnData file containing labeled reference samples.
        query_path: AnnData file containing query samples.

    Returns:
        Aligned reference and query AnnData objects.

    Raises:
        FileNotFoundError: If either input file does not exist.
        KeyError: If required observation columns are unavailable.
    """
    reference_file = Path(reference_path).expanduser().resolve()
    query_file = Path(query_path).expanduser().resolve()

    # Validate both inputs before loading them into memory.
    for input_file in (reference_file, query_file):
        if not input_file.is_file():
            raise FileNotFoundError(input_file)

    # Standardize required observation fields before gene alignment.
    reference_adata = standardize_obs_columns(reference_file)
    query_adata = standardize_obs_columns(query_file, allow_missing_label=True)

    # Use the shared alignment utility so both matrices have identical genes and order.
    aligned_reference, aligned_query = align_adatas_to_union_genes(
        [reference_adata, query_adata]
    )
    return aligned_reference, aligned_query


def prepare_cross_dataset_workflow(
    reference_h5ad: PathLike = DEFAULT_REFERENCE_PATH,
    query_h5ad: PathLike = DEFAULT_QUERY_PATH,
    output_dir: PathLike = DEFAULT_OUTPUT_DIRECTORY,
    train_samples: Iterable[str] = ("train_sample_1", "train_sample_2"),
    validation_samples: Iterable[str] = ("validation_sample",),
    query_samples: Iterable[str] = ("query_sample",),
    communication_graph_neighbors: int = 2,
    force: bool = False,
) -> Dict[str, object]:
    """Prepare generic cross-dataset graph inputs for training and prediction.

    Args:
        reference_h5ad: Labeled reference AnnData file.
        query_h5ad: Query AnnData file.
        output_dir: Directory receiving intermediate and graph files.
        train_samples: Reference sample identifiers used for training.
        validation_samples: Reference sample identifiers used for validation.
        query_samples: Query sample identifiers used for prediction.
        communication_graph_neighbors: Neighbor count for communication graphs.
        force: Overwrite files that already exist.

    Returns:
        Dictionary of paths returned by ``prepare_data_workflow``.
    """
    # Delegate normalization, splitting, pathway features, and graph construction.
    return prepare_data_workflow(
        train_h5ad=reference_h5ad,
        query_h5ad=query_h5ad,
        output_dir=output_dir,
        train_batches=[str(sample) for sample in train_samples],
        valid_batches=[str(sample) for sample in validation_samples],
        query_batches=[str(sample) for sample in query_samples],
        cgraph_k=communication_graph_neighbors,
        train_output_name="reference_data",
        query_output_name="query_data",
        force=force,
    )
