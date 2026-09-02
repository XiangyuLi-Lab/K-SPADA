"""Prepare a generic reference-to-query transfer dataset for K-SPADA."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import anndata as ad

from . import prepare_data_workflow
from .binning import bin_adata


PathLike = Union[str, Path]
DEFAULT_DATA_DIRECTORY = Path("/your_path/example_dataset")
DEFAULT_SOURCE_PATH = DEFAULT_DATA_DIRECTORY / "source_data.h5ad"
DEFAULT_BINNED_PATH = DEFAULT_DATA_DIRECTORY / "binned_data.h5ad"
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_DATA_DIRECTORY / "prepared_data"


def prepare_binned_source(
    source_path: PathLike = DEFAULT_SOURCE_PATH,
    output_path: PathLike = DEFAULT_BINNED_PATH,
    bin_count: int = 8,
    random_seed: int = 1,
    force: bool = False,
) -> Path:
    """Bin a generic AnnData expression matrix and write the processed file.

    Args:
        source_path: Input AnnData file containing the complete dataset.
        output_path: Destination for the binned AnnData file.
        bin_count: Total number of bins, including zero as bin 0.
        random_seed: Seed used to resolve values located exactly on bin edges.
        force: Overwrite the destination when it already exists.

    Returns:
        Path to the written binned AnnData file.

    Raises:
        FileNotFoundError: If the source AnnData file does not exist.
        FileExistsError: If the destination exists and ``force`` is false.
    """
    source_file = Path(source_path).expanduser().resolve()
    destination_file = Path(output_path).expanduser().resolve()

    # Check input and output paths before loading the expression matrix.
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    if destination_file.exists() and not force:
        raise FileExistsError(destination_file)

    # Apply the shared zero-preserving quantile-binning implementation.
    source_adata = ad.read_h5ad(source_file)
    binned_adata = bin_adata(
        source_adata,
        bin_count=bin_count,
        random_seed=random_seed,
        copy=True,
    )

    # Create the destination directory only when the processed file is written.
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    binned_adata.write_h5ad(destination_file)
    return destination_file


def prepare_transfer_workflow(
    reference_h5ad: PathLike,
    query_h5ad: PathLike,
    output_dir: PathLike = DEFAULT_OUTPUT_DIRECTORY,
    train_samples: Iterable[str] = ("train_sample_1", "train_sample_2"),
    validation_samples: Iterable[str] = ("validation_sample",),
    query_samples: Iterable[str] = ("query_sample",),
    communication_graph_neighbors: int = 2,
    pathway_file: Optional[PathLike] = None,
    force: bool = False,
) -> Dict[str, object]:
    """Build generic reference, validation, and query graph files.

    Args:
        reference_h5ad: AnnData file containing training and validation samples.
        query_h5ad: AnnData file containing query samples.
        output_dir: Directory receiving intermediate and graph files.
        train_samples: Sample identifiers used for model fitting.
        validation_samples: Sample identifiers used during training validation.
        query_samples: Sample identifiers used for prediction.
        communication_graph_neighbors: Neighbor count for communication graphs.
        pathway_file: Optional pathway definition file.
        force: Overwrite files that already exist.

    Returns:
        Dictionary of paths returned by ``prepare_data_workflow``.
    """
    # Reuse the package-level preparation workflow instead of duplicating graph logic.
    return prepare_data_workflow(
        train_h5ad=reference_h5ad,
        query_h5ad=query_h5ad,
        output_dir=output_dir,
        train_batches=[str(sample) for sample in train_samples],
        valid_batches=[str(sample) for sample in validation_samples],
        query_batches=[str(sample) for sample in query_samples],
        cgraph_k=communication_graph_neighbors,
        pathway_file=pathway_file,
        train_output_name="reference_data",
        query_output_name="query_data",
        force=force,
    )
