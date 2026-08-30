"""Prepare no-binning Fly3d data for cross-time K-SPADA prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_FLY_DIR = Path("/home/lixiangyu/TOSICA-main/data/Fly3d")
DEFAULT_E16_HVG = DEFAULT_FLY_DIR / "fly_E16-18h_hvg_3000.h5ad"
DEFAULT_E14_HVG = DEFAULT_FLY_DIR / "fly_E14-16h_hvg_3000.h5ad"

TRAIN_BATCH_PREFIX = DEFAULT_FLY_DIR / "fly_E16-18h_hvg_3000_train_query_"
QUERY_BATCH_PREFIX = DEFAULT_FLY_DIR / "fly_E14-16h_hvg_3000_span_query_"
TRAIN_VALID_H5AD = DEFAULT_FLY_DIR / "fly_E16-18h_hvg_3000_all_real_train_plus_valid_copy_901.h5ad"
TRAIN_VALID_OUTPUT_PREFIX = DEFAULT_FLY_DIR / "fly_E16-18h_hvg_3000_all_real_train_valid_copy_901"
MANIFEST_JSON = DEFAULT_FLY_DIR / "fly_hvg_no_binning_all16_to_all14_manifest.json"
MANIFEST_CSV = DEFAULT_FLY_DIR / "fly_hvg_no_binning_all16_to_all14_manifest.csv"

E16_TRAIN_BATCHES = [str(i) for i in range(1, 15)]
E14_QUERY_BATCHES = [str(i) for i in range(1, 17)]
VALID_COPY_SOURCE_BATCH = "14"
VALID_COPY_BATCH = "901"


def parse_batch_list(batch_text: str) -> List[str]:
    """Parse a comma-separated batch list."""
    return [item.strip() for item in batch_text.split(",") if item.strip()]


def as_csr_matrix(matrix) -> sparse.csr_matrix:
    """Convert an expression matrix to CSR format."""
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def assert_required_obs(adata: ad.AnnData, adata_path: Path) -> None:
    """Validate observation columns required by K-SPADA."""
    required_columns = ["x", "y", "Celltype", "batch"]
    missing_columns = [column for column in required_columns if column not in adata.obs.columns]
    if missing_columns:
        raise KeyError(f"{adata_path} missing obs columns: {missing_columns}")


def assert_same_gene_space(train_adata: ad.AnnData, query_adata: ad.AnnData) -> None:
    """Verify that training and query data use the same ordered genes."""
    train_genes = list(train_adata.var_names.astype(str))
    query_genes = list(query_adata.var_names.astype(str))
    if train_genes != query_genes:
        raise ValueError("E16-18h and E14-16h use different ordered genes and cannot be used for direct cross-time prediction.")


def load_hvg_adata(adata_path: Path, label: str) -> ad.AnnData:
    """Load an unbinned HVG AnnData object and validate its fields."""
    print(f"[load] {label}: {adata_path}")
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing h5ad: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    assert_required_obs(adata, adata_path)
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    print(f"[load] {label}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"[load] {label} batches: {sorted(adata.obs['batch'].unique().tolist(), key=lambda x: int(x))}")
    return adata


def write_single_batch_files(adata: ad.AnnData, batch_ids: Iterable[str], output_prefix: Path, force: bool) -> None:
    """Write one AnnData file for each requested batch."""
    for batch_id in batch_ids:
        output_path = Path(f"{output_prefix}{batch_id}.h5ad")
        if output_path.exists() and not force:
            print(f"[skip] single batch exists: {output_path}")
            continue
        batch_mask = adata.obs["batch"].astype(str) == str(batch_id)
        if int(batch_mask.sum()) == 0:
            raise ValueError(f"No cells found for batch={batch_id}")
        batch_adata = adata[batch_mask].copy()
        batch_adata.uns["binning"] = "not_performed"
        batch_adata.uns["source_matrix"] = "hvg_3000_no_binning"
        print(f"[write] batch {batch_id}: {batch_adata.n_obs} cells -> {output_path}")
        batch_adata.write_h5ad(output_path)


def create_valid_copy_file(train_adata: ad.AnnData, source_batch: str, valid_batch: str, output_prefix: Path, force: bool) -> Path:
    """Create a validation-compatible copy of one training batch."""
    output_path = Path(f"{output_prefix}{valid_batch}.h5ad")
    if output_path.exists() and not force:
        print(f"[skip] valid copy exists: {output_path}")
        return output_path

    source_mask = train_adata.obs["batch"].astype(str) == str(source_batch)
    if int(source_mask.sum()) == 0:
        raise ValueError(f"No cells found for valid copy source batch={source_batch}")

    valid_adata = train_adata[source_mask].copy()
    valid_adata.obs_names = [f"valid_copy_{valid_batch}:{cell_name}" for cell_name in valid_adata.obs_names.astype(str)]
    valid_adata.obs["batch"] = str(valid_batch)
    valid_adata.obs["valid_copy_source_batch"] = str(source_batch)
    valid_adata.uns["binning"] = "not_performed"
    valid_adata.uns["purpose"] = "validation_copy_for_unmodified_TOSICA_train"
    print(f"[write] valid copy batch {valid_batch} from source batch {source_batch}: {valid_adata.n_obs} cells -> {output_path}")
    valid_adata.write_h5ad(output_path)
    return output_path


def copy_valid_copy_path_scores_and_graph(source_batch: str, valid_batch: str, output_prefix: Path, graph_k: int, force: bool) -> None:
    """Copy pathway scores and graph data for the validation-compatible batch."""
    valid_h5ad = Path(f"{output_prefix}{valid_batch}.h5ad")
    if not valid_h5ad.exists():
        raise FileNotFoundError(f"Missing valid copy h5ad: {valid_h5ad}")

    valid_adata = sc.read_h5ad(valid_h5ad)
    source_and_target_paths = [
        (
            Path(f"{output_prefix}{source_batch}_path_scores.h5ad"),
            Path(f"{output_prefix}{valid_batch}_path_scores.h5ad"),
        ),
        (
            Path(f"{output_prefix}{source_batch}_path_scores_graph_k_{graph_k}.h5ad"),
            Path(f"{output_prefix}{valid_batch}_path_scores_graph_k_{graph_k}.h5ad"),
        ),
    ]

    for source_path, target_path in source_and_target_paths:
        if target_path.exists() and not force:
            print(f"[skip] valid copy graph artifact exists: {target_path}")
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source graph artifact for valid copy: {source_path}")

        copied_adata = sc.read_h5ad(source_path)
        if copied_adata.n_obs != valid_adata.n_obs:
            raise ValueError(f"Cannot copy {source_path.name}: source cells={copied_adata.n_obs}, valid cells={valid_adata.n_obs}")
        copied_adata.obs = valid_adata.obs.copy()
        copied_adata.uns["valid_copy_source_batch"] = str(source_batch)
        copied_adata.uns["purpose"] = "validation_copy_for_unmodified_TOSICA_train"
        print(f"[copy] valid copy graph artifact: {source_path} -> {target_path}")
        copied_adata.write_h5ad(target_path)


def merge_train_with_valid_copy(train_adata: ad.AnnData, source_batch: str, valid_batch: str, output_path: Path, force: bool) -> Path:
    """Merge all training cells with the validation-compatible batch."""
    if output_path.exists() and not force:
        print(f"[skip] train_valid input exists: {output_path}")
        return output_path

    train_real = train_adata.copy()
    train_real.obs["batch"] = train_real.obs["batch"].astype(str)
    train_real.obs["query_span_split"] = "train_real"
    train_real.uns["binning"] = "not_performed"

    source_mask = train_adata.obs["batch"].astype(str) == str(source_batch)
    valid_copy = train_adata[source_mask].copy()
    valid_copy.obs_names = [f"valid_copy_{valid_batch}:{cell_name}" for cell_name in valid_copy.obs_names.astype(str)]
    valid_copy.obs["batch"] = str(valid_batch)
    valid_copy.obs["query_span_split"] = "valid_copy"
    valid_copy.obs["valid_copy_source_batch"] = str(source_batch)

    merged_x = sparse.vstack([as_csr_matrix(train_real.X), as_csr_matrix(valid_copy.X)]).tocsr()
    merged_obs = pd.concat([train_real.obs.copy(), valid_copy.obs.copy()], axis=0)
    merged_var = train_real.var.copy()
    merged_adata = ad.AnnData(X=merged_x, obs=merged_obs, var=merged_var)
    merged_adata.uns["binning"] = "not_performed"
    merged_adata.uns["train_real_batches"] = E16_TRAIN_BATCHES
    merged_adata.uns["valid_copy_batch"] = str(valid_batch)
    merged_adata.uns["valid_copy_source_batch"] = str(source_batch)
    merged_adata.uns["design"] = "all E16-18h real batches are train; valid copy only keeps unmodified train.py runnable"

    print(f"[write] train_valid input: {merged_adata.n_obs} cells x {merged_adata.n_vars} genes -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_adata.write_h5ad(output_path)
    return output_path


def load_cgraph_path_data() -> Dict[str, List[str]]:
    """Load the default ligand-receptor pathway data."""
    from ..createCGraph import read_pathway_data

    return read_pathway_data()


def create_path_scores_compatible_with_sparse_x(input_h5ad: Path, output_prefix: Path) -> None:
    """Calculate pathway scores while supporting sparse expression matrices."""
    from ..createCGraph import calculatePathEnrichment

    path_data = load_cgraph_path_data()
    adata = sc.read_h5ad(input_h5ad)
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32, copy=False)
    updated_adata = calculatePathEnrichment(adata, path_data)
    non_zero_elements = np.count_nonzero(updated_adata.obsm["path_scores"])
    output_h5ad = Path(f"{output_prefix}_path_scores.h5ad")
    print(f"[path] {input_h5ad.name}: non-zero path_scores={non_zero_elements}")
    updated_adata.write_h5ad(output_h5ad)


def has_matching_var_names(reference_h5ad: Path, candidate_h5ad: Path) -> bool:
    """Check whether two AnnData files use identical ordered genes."""
    reference_adata = ad.read_h5ad(reference_h5ad, backed="r")
    candidate_adata = ad.read_h5ad(candidate_h5ad, backed="r")
    try:
        return reference_adata.var_names.equals(candidate_adata.var_names)
    finally:
        reference_adata.file.close()
        candidate_adata.file.close()


def build_path_scores_and_graph(batch_ids: Iterable[str], file_prefix: Path, graph_k: int, force: bool) -> None:
    """Build pathway scores and communication graphs for each batch."""
    from ..createCGraph import getCommunicateGraph

    for batch_id in batch_ids:
        input_h5ad = Path(f"{file_prefix}{batch_id}.h5ad")
        path_scores_h5ad = Path(f"{file_prefix}{batch_id}_path_scores.h5ad")
        graph_h5ad = Path(f"{file_prefix}{batch_id}_path_scores_graph_k_{graph_k}.h5ad")

        if not input_h5ad.exists():
            raise FileNotFoundError(f"Missing single batch h5ad: {input_h5ad}")

        reuse_path_scores = path_scores_h5ad.exists() and not force and has_matching_var_names(input_h5ad, path_scores_h5ad)
        if reuse_path_scores:
            print(f"[skip] path_scores exists: {path_scores_h5ad}")
        else:
            if path_scores_h5ad.exists() and not force:
                print(f"[rebuild] stale path_scores gene space: {path_scores_h5ad}")
            create_path_scores_compatible_with_sparse_x(input_h5ad, Path(f"{file_prefix}{batch_id}"))

        reuse_graph = graph_h5ad.exists() and not force and has_matching_var_names(input_h5ad, graph_h5ad)
        if reuse_graph:
            print(f"[skip] graph exists: {graph_h5ad}")
        else:
            if graph_h5ad.exists() and not force:
                print(f"[rebuild] stale graph gene space: {graph_h5ad}")
            print(f"[graph] batch {batch_id}: k={graph_k}")
            graph_adata = getCommunicateGraph(str(path_scores_h5ad), k=graph_k)
            graph_adata.write_h5ad(graph_h5ad)


def create_train_valid_cgraph(train_valid_h5ad: Path, train_batches: List[str], valid_batches: List[str], graph_k: int, force: bool) -> Path:
    """Create the training and validation communication graph file."""
    from ..createCGraph import CreateTrainData

    output_h5ad = Path(f"{TRAIN_VALID_OUTPUT_PREFIX}_train_valid_CGraph_k_{graph_k}.h5ad")
    if output_h5ad.exists() and not force:
        print(f"[skip] train_valid_CGraph exists: {output_h5ad}")
        return output_h5ad

    print(f"[cgraph] train batches: {train_batches}")
    print(f"[cgraph] valid batches: {valid_batches}")
    print(f"[cgraph] train_valid input: {train_valid_h5ad}")
    CreateTrainData(
        train_batch=train_batches,
        valid_batch=valid_batches,
        file_pre=str(TRAIN_BATCH_PREFIX),
        train_adata_name=str(train_valid_h5ad),
        file_name=str(TRAIN_VALID_OUTPUT_PREFIX),
        k=graph_k,
    )
    return output_h5ad


def write_manifest(train_cgraph_h5ad: Path, graph_k: int) -> None:
    """Write the generated training and query file manifest."""
    query_graph_paths = {
        batch_id: str(Path(f"{QUERY_BATCH_PREFIX}{batch_id}_path_scores_graph_k_{graph_k}.h5ad"))
        for batch_id in E14_QUERY_BATCHES
    }
    manifest = {
        "design": "E16-18h all real batches train; E14-16h all real batches query; no binning",
        "train_h5ad_for_TOSICA_train": str(train_cgraph_h5ad),
        "train_batches_for_data_batch": E16_TRAIN_BATCHES,
        "valid_batches_inside_training_file": [VALID_COPY_BATCH],
        "valid_copy_source_batch": VALID_COPY_SOURCE_BATCH,
        "query_batches_for_prediction": E14_QUERY_BATCHES,
        "query_graph_h5ads": query_graph_paths,
        "graph_k": graph_k,
        "binning": "not_performed",
    }

    with MANIFEST_JSON.open("w") as file_handle:
        json.dump(manifest, file_handle, indent=2)

    rows = []
    for batch_id, query_path in query_graph_paths.items():
        rows.append(
            {
                "query_batch": batch_id,
                "query_graph_h5ad": query_path,
                "train_h5ad_for_TOSICA_train": str(train_cgraph_h5ad),
                "train_batches_for_data_batch": ",".join(E16_TRAIN_BATCHES),
                "graph_k": graph_k,
                "binning": "not_performed",
            }
        )
    pd.DataFrame(rows).to_csv(MANIFEST_CSV, index=False)
    print(f"[manifest] {MANIFEST_JSON}")
    print(f"[manifest] {MANIFEST_CSV}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Fly3d data preparation."""
    parser = argparse.ArgumentParser(description="Prepare Fly3d hvg_3000 no-binning all E16-18h train to all E14-16h query CGraph data.")
    parser.add_argument("--graph-k", type=int, default=3, help="Communication-graph neighbor count.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing intermediate and output files.")
    parser.add_argument("--skip-graph", action="store_true", help="Write batch files and train-validation input without building pathway scores or graphs.")
    return parser.parse_args()


def main() -> None:
    """Run the complete Fly3d no-binning data preparation workflow."""
    args = parse_args()
    graph_k = int(args.graph_k)
    train_adata = load_hvg_adata(DEFAULT_E16_HVG, "E16-18h train source")
    query_adata = load_hvg_adata(DEFAULT_E14_HVG, "E14-16h query source")
    assert_same_gene_space(train_adata, query_adata)

    write_single_batch_files(train_adata, E16_TRAIN_BATCHES, TRAIN_BATCH_PREFIX, force=args.force)
    write_single_batch_files(query_adata, E14_QUERY_BATCHES, QUERY_BATCH_PREFIX, force=args.force)
    create_valid_copy_file(train_adata, VALID_COPY_SOURCE_BATCH, VALID_COPY_BATCH, TRAIN_BATCH_PREFIX, force=args.force)
    train_valid_h5ad = merge_train_with_valid_copy(
        train_adata=train_adata,
        source_batch=VALID_COPY_SOURCE_BATCH,
        valid_batch=VALID_COPY_BATCH,
        output_path=TRAIN_VALID_H5AD,
        force=args.force,
    )

    if args.skip_graph:
        print("[done] skip graph enabled; graph files were not generated.")
        return

    build_path_scores_and_graph(E16_TRAIN_BATCHES, TRAIN_BATCH_PREFIX, graph_k=graph_k, force=args.force)
    copy_valid_copy_path_scores_and_graph(VALID_COPY_SOURCE_BATCH, VALID_COPY_BATCH, TRAIN_BATCH_PREFIX, graph_k=graph_k, force=args.force)
    build_path_scores_and_graph(E14_QUERY_BATCHES, QUERY_BATCH_PREFIX, graph_k=graph_k, force=args.force)
    train_cgraph_h5ad = create_train_valid_cgraph(
        train_valid_h5ad=train_valid_h5ad,
        train_batches=E16_TRAIN_BATCHES,
        valid_batches=[VALID_COPY_BATCH],
        graph_k=graph_k,
        force=args.force,
    )
    write_manifest(train_cgraph_h5ad, graph_k=graph_k)
    print("[done] Fly3d no-binning all16-to-all14 CGraph data prepared.")


if __name__ == "__main__":
    main()
