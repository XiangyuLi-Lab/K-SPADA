"""Data preparation utilities used by the K-SPADA tutorials."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from ..createCGraph import create_train_valid_cgraph, prepare_query_cgraph
from .binning import bin_adata, bin_expression_matrix, bin_expression_row, digitize_like_scgpt, to_dense_float32

AnnDataLike = Union[str, Path, ad.AnnData]


def read_adata(adata_input: AnnDataLike, copy: bool = True) -> ad.AnnData:
    """Read an AnnData object or load one from a .h5ad path.

    Args:
        adata_input: AnnData object or .h5ad path.
        copy: Whether to copy an in-memory AnnData object.

    Returns:
        AnnData: Loaded or copied AnnData object.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if isinstance(adata_input, ad.AnnData):
        return adata_input.copy() if copy else adata_input
    if not isinstance(adata_input, (str, Path)):
        raise TypeError("adata_input must be an AnnData object or local h5ad path")
    input_path = Path(adata_input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return ad.read_h5ad(input_path)


def use_count_layer(
    adata_input: AnnDataLike,
    layer_name: str = "raw_counts",
    copy: bool = True,
) -> ad.AnnData:
    """Move a count layer into adata.X for downstream normalization.

    Args:
        adata_input: AnnData object or .h5ad path.
        layer_name: Layer used as the expression matrix.
        copy: Whether to return a new AnnData object.

    Returns:
        AnnData: AnnData whose X matrix comes from the requested layer.

    Raises:
        KeyError: If the requested layer is missing.
    """
    adata_object = read_adata(adata_input, copy=copy)
    if layer_name not in adata_object.layers:
        raise KeyError(f"Missing adata.layers['{layer_name}']")
    adata_object.X = adata_object.layers[layer_name].copy()
    return adata_object


def standardize_obs_columns(
    adata_input: AnnDataLike,
    x_key: str = "raw_x",
    y_key: str = "raw_y",
    batch_key: str = "slice_ID",
    label_key: str = "annotation",
    output_x_key: str = "x",
    output_y_key: str = "y",
    output_batch_key: str = "batch",
    output_label_key: str = "Celltype",
    batch_mapper: Optional[Callable[[object], object]] = None,
    allow_missing_label: bool = False,
    copy: bool = True,
) -> ad.AnnData:
    """Copy dataset-specific obs columns into the names expected by KSPADA.

    Args:
        adata_input: AnnData object or .h5ad path.
        x_key: Source x-coordinate column.
        y_key: Source y-coordinate column.
        batch_key: Source batch/slice column.
        label_key: Source label column.
        output_x_key: Output x-coordinate column expected by graph code.
        output_y_key: Output y-coordinate column expected by graph code.
        output_batch_key: Output batch column expected by graph code.
        output_label_key: Output label column expected by training code.
        batch_mapper: Optional function for converting raw batch values.
        allow_missing_label: Whether to fill output_label_key with Unknown when label_key is missing.
        copy: Whether to return a new AnnData object.

    Returns:
        AnnData: AnnData with standardized obs columns.

    Raises:
        KeyError: If any required source column is missing.
    """
    adata_object = read_adata(adata_input, copy=copy)
    required_columns = [x_key, y_key, batch_key]
    if not allow_missing_label:
        required_columns.append(label_key)
    missing_columns = [column for column in required_columns if column not in adata_object.obs]
    if missing_columns:
        raise KeyError(f"Missing obs columns: {missing_columns}")
    adata_object.obs[output_x_key] = adata_object.obs[x_key].values
    adata_object.obs[output_y_key] = adata_object.obs[y_key].values
    raw_batch_values = adata_object.obs[batch_key]
    if batch_mapper is None:
        adata_object.obs[output_batch_key] = raw_batch_values.astype(str).values
    else:
        adata_object.obs[output_batch_key] = raw_batch_values.map(batch_mapper).astype(str).values
    if label_key in adata_object.obs:
        adata_object.obs[output_label_key] = adata_object.obs[label_key].astype(str).values
    elif allow_missing_label:
        adata_object.obs[output_label_key] = "Unknown"
    return adata_object


def build_union_genes(adatas: Sequence[ad.AnnData]) -> pd.Index:
    """Build a shared gene index from multiple AnnData objects.

    Args:
        adatas: AnnData objects to align.

    Returns:
        pandas.Index: Union of all var_names.

    Raises:
        ValueError: If no AnnData objects are supplied.
    """
    if not adatas:
        raise ValueError("At least one AnnData object is required")
    union_genes = pd.Index(adatas[0].var_names)
    for adata_object in adatas[1:]:
        union_genes = union_genes.union(pd.Index(adata_object.var_names))
    return union_genes


def align_adata_to_genes(
    adata_input: AnnDataLike,
    target_genes: Sequence[str],
    copy: bool = True,
) -> ad.AnnData:
    """Align AnnData to a target gene order and fill missing genes with zero.

    Args:
        adata_input: AnnData object or .h5ad path.
        target_genes: Target gene names and order.
        copy: Whether to copy an in-memory AnnData object before alignment.

    Returns:
        AnnData: Aligned AnnData with exactly target_genes as var_names.
    """
    adata_object = read_adata(adata_input, copy=copy)
    target_gene_index = pd.Index([str(gene) for gene in target_genes])
    source_gene_index = pd.Index([str(gene) for gene in adata_object.var_names])
    source_positions = source_gene_index.get_indexer(target_gene_index)
    output_positions = np.where(source_positions >= 0)[0]
    existing_source_positions = source_positions[output_positions]
    if sparse.issparse(adata_object.X):
        source_matrix = adata_object.X[:, existing_source_positions].tocoo()
        aligned_columns = output_positions[source_matrix.col]
        aligned_matrix = sparse.csr_matrix(
            (source_matrix.data, (source_matrix.row, aligned_columns)),
            shape=(adata_object.n_obs, len(target_gene_index)),
        )
    else:
        aligned_matrix = np.zeros((adata_object.n_obs, len(target_gene_index)), dtype=np.asarray(adata_object.X).dtype)
        aligned_matrix[:, output_positions] = np.asarray(adata_object.X)[:, existing_source_positions]
    aligned_var = pd.DataFrame(index=target_gene_index)
    aligned_adata = ad.AnnData(
        X=aligned_matrix,
        obs=adata_object.obs.copy(),
        var=aligned_var,
        uns=adata_object.uns.copy(),
        obsm=adata_object.obsm.copy(),
    )
    return aligned_adata


def align_adatas_to_union_genes(adatas: Sequence[AnnDataLike]) -> list:
    """Align multiple AnnData objects to the same union gene space.

    Args:
        adatas: AnnData objects or .h5ad paths.

    Returns:
        list: Aligned AnnData objects in the same order as input.
    """
    loaded_adatas = [read_adata(adata_input, copy=True) for adata_input in adatas]
    union_genes = build_union_genes(loaded_adatas)
    return [align_adata_to_genes(adata_object, union_genes, copy=False) for adata_object in loaded_adatas]


def normalize_total_log1p(
    adata_input: AnnDataLike,
    target_sum: float = 1e4,
    copy: bool = True,
) -> ad.AnnData:
    """Run scanpy normalize_total followed by log1p.

    Args:
        adata_input: AnnData object or .h5ad path.
        target_sum: Target library size for normalize_total.
        copy: Whether to return a new AnnData object.

    Returns:
        AnnData: Normalized AnnData.
    """
    adata_object = read_adata(adata_input, copy=copy)
    sc.pp.normalize_total(adata_object, target_sum=target_sum)
    sc.pp.log1p(adata_object)
    return adata_object


def split_by_batch(
    adata_input: AnnDataLike,
    batch_ids: Optional[Iterable[Union[str, int]]] = None,
    batch_key: str = "batch",
    output_prefix: Optional[Union[str, Path]] = None,
    force: bool = False,
) -> dict:
    """Split AnnData by batch and optionally write one .h5ad per batch.

    Args:
        adata_input: AnnData object or .h5ad path.
        batch_ids: Batch IDs to keep. If None, all batches are used.
        batch_key: Column in obs containing batch IDs.
        output_prefix: Optional output prefix. Files are written as {prefix}{batch_id}.h5ad.
        force: Overwrite existing files when output_prefix is provided.

    Returns:
        dict: Mapping from batch ID to split AnnData or written file path.

    Raises:
        KeyError: If batch_key is missing.
    """
    adata_object = read_adata(adata_input, copy=False)
    if batch_key not in adata_object.obs:
        raise KeyError(f"Missing adata.obs['{batch_key}']")
    if batch_ids is None:
        selected_batches = [str(value) for value in sorted(adata_object.obs[batch_key].astype(str).unique())]
    else:
        selected_batches = [str(batch_id) for batch_id in batch_ids]
    output_items = {}
    output_prefix_path = Path(output_prefix).expanduser() if output_prefix is not None else None
    if output_prefix_path is not None:
        output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    batch_values = adata_object.obs[batch_key].astype(str)
    for batch_id in selected_batches:
        batch_adata = adata_object[batch_values == batch_id].copy()
        if output_prefix_path is None:
            output_items[batch_id] = batch_adata
        else:
            output_path = output_prefix_path.parent / f"{output_prefix_path.name}{batch_id}.h5ad"
            if force or not output_path.exists():
                batch_adata.write_h5ad(output_path)
            output_items[batch_id] = output_path
    return output_items


def write_adata(adata_input: AnnDataLike, output_path: Union[str, Path], force: bool = False) -> Path:
    """Write AnnData to disk with a small overwrite guard.

    Args:
        adata_input: AnnData object or .h5ad path.
        output_path: Output .h5ad path.
        force: Overwrite if the file exists.

    Returns:
        Path: Written output path.

    Raises:
        FileExistsError: If output exists and force is False.
    """
    resolved_output_path = Path(output_path).expanduser().resolve()
    if resolved_output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {resolved_output_path}")
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_object = read_adata(adata_input, copy=False)
    adata_object.write_h5ad(resolved_output_path)
    return resolved_output_path


def prepare_data_workflow(
    train_h5ad: Union[str, Path],
    query_h5ad: Union[str, Path],
    output_dir: Union[str, Path],
    train_batches: Iterable[Union[str, int]],
    valid_batches: Iterable[Union[str, int]],
    query_batches: Optional[Iterable[Union[str, int]]] = None,
    count_layer: Optional[str] = None,
    x_key: str = "x",
    y_key: str = "y",
    batch_key: str = "batch",
    label_key: str = "Celltype",
    batch_mapper: Optional[Callable[[object], object]] = None,
    target_sum: float = 1e4,
    cgraph_k: int = 6,
    pathway_file: Optional[Union[str, Path]] = None,
    train_output_name: str = "train_norm",
    query_output_name: str = "query_norm",
    force: bool = False,
) -> dict:
    """Run one complete preprocessing workflow from raw h5ad files to KSPADA-ready CGraph files.

    Args:
        train_h5ad: Training .h5ad path.
        query_h5ad: Query .h5ad path.
        output_dir: Directory for normalized, split, and CGraph h5ad files.
        train_batches: Batch IDs used for model training.
        valid_batches: Batch IDs used for model validation.
        query_batches: Query batch IDs. If None, all query batches are used.
        count_layer: Optional layer name copied into X before normalization, such as raw_counts.
        x_key: Source x-coordinate column in obs.
        y_key: Source y-coordinate column in obs.
        batch_key: Source batch column in obs.
        label_key: Source label column in obs.
        batch_mapper: Optional function used to convert raw batch values to model batch IDs.
        target_sum: Library-size target for scanpy normalize_total.
        cgraph_k: K used to build communication graphs.
        pathway_file: Optional pathway file. Defaults to packaged KSPADA/resources/output_path4.txt.
        train_output_name: File prefix for normalized training data.
        query_output_name: File prefix for normalized query data.
        force: Overwrite existing files.

    Returns:
        dict: Key output paths and batch IDs.
    """
    output_directory = Path(output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if count_layer is None:
        train_adata = standardize_obs_columns(
            train_h5ad,
            x_key=x_key,
            y_key=y_key,
            batch_key=batch_key,
            label_key=label_key,
            batch_mapper=batch_mapper,
        )
        query_adata = standardize_obs_columns(
            query_h5ad,
            x_key=x_key,
            y_key=y_key,
            batch_key=batch_key,
            label_key=label_key,
            batch_mapper=batch_mapper,
            allow_missing_label=True,
        )
    else:
        train_adata = use_count_layer(train_h5ad, layer_name=count_layer)
        query_adata = use_count_layer(query_h5ad, layer_name=count_layer)
        train_adata = standardize_obs_columns(
            train_adata,
            x_key=x_key,
            y_key=y_key,
            batch_key=batch_key,
            label_key=label_key,
            batch_mapper=batch_mapper,
            copy=False,
        )
        query_adata = standardize_obs_columns(
            query_adata,
            x_key=x_key,
            y_key=y_key,
            batch_key=batch_key,
            label_key=label_key,
            batch_mapper=batch_mapper,
            allow_missing_label=True,
            copy=False,
        )
    aligned_train_adata, aligned_query_adata = align_adatas_to_union_genes([train_adata, query_adata])
    normalized_train_adata = normalize_total_log1p(aligned_train_adata, target_sum=target_sum, copy=False)
    normalized_query_adata = normalize_total_log1p(aligned_query_adata, target_sum=target_sum, copy=False)
    train_norm_path = output_directory / f"{train_output_name}.h5ad"
    query_norm_path = output_directory / f"{query_output_name}.h5ad"
    write_adata(normalized_train_adata, train_norm_path, force=force)
    write_adata(normalized_query_adata, query_norm_path, force=force)
    train_batch_list = [str(batch_id) for batch_id in train_batches]
    valid_batch_list = [str(batch_id) for batch_id in valid_batches]
    if query_batches is None:
        query_batch_list = [str(batch_id) for batch_id in sorted(normalized_query_adata.obs["batch"].astype(str).unique())]
    else:
        query_batch_list = [str(batch_id) for batch_id in query_batches]
    train_split_prefix = output_directory / f"{train_output_name}_batch_"
    query_split_prefix = output_directory / f"{query_output_name}_batch_"
    train_split_paths = split_by_batch(
        normalized_train_adata,
        batch_ids=sorted(set(train_batch_list + valid_batch_list)),
        output_prefix=train_split_prefix,
        force=force,
    )
    query_split_paths = split_by_batch(
        normalized_query_adata,
        batch_ids=query_batch_list,
        output_prefix=query_split_prefix,
        force=force,
    )
    graph_train_paths = {}
    graph_query_paths = {}
    for batch_id, split_path in train_split_paths.items():
        graph_output_path = Path(str(split_path).replace(".h5ad", f"_path_scores_graph_k_{cgraph_k}.h5ad"))
        graph_train_paths[batch_id] = prepare_query_cgraph(
            split_path,
            graph_output_path,
            pathway_file=pathway_file,
            k=cgraph_k,
            force=force,
        )
    for batch_id, split_path in query_split_paths.items():
        graph_output_path = Path(str(split_path).replace(".h5ad", f"_path_scores_graph_k_{cgraph_k}.h5ad"))
        graph_query_paths[batch_id] = prepare_query_cgraph(
            split_path,
            graph_output_path,
            pathway_file=pathway_file,
            k=cgraph_k,
            force=force,
        )
    train_valid_prefix = output_directory / f"{train_output_name}_train_{'_'.join(train_batch_list)}_valid_{'_'.join(valid_batch_list)}"
    train_valid_h5ad = create_train_valid_cgraph(
        train_batches=train_batch_list,
        valid_batches=valid_batch_list,
        batch_file_prefix=train_split_prefix,
        train_adata_path=train_norm_path,
        output_prefix=train_valid_prefix,
        k=cgraph_k,
        force=force,
    )
    return {
        "train_norm_path": train_norm_path,
        "query_norm_path": query_norm_path,
        "train_split_prefix": train_split_prefix,
        "query_split_prefix": query_split_prefix,
        "train_batches": train_batch_list,
        "valid_batches": valid_batch_list,
        "query_batches": query_batch_list,
        "train_split_paths": train_split_paths,
        "query_split_paths": query_split_paths,
        "train_graph_paths": graph_train_paths,
        "query_graph_paths": graph_query_paths,
        "train_valid_h5ad": train_valid_h5ad,
    }


# Import generic, higher-level workflows after the shared utilities are defined.
from .cross_dataset import load_and_align_datasets, prepare_cross_dataset_workflow
from .transfer_dataset import prepare_binned_source, prepare_transfer_workflow
