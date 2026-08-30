"""Prepare binning_8_with_0 liver cancer transfer datasets for K-SPADA."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import warnings
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path('/home/lixiangyu/TOSICA-main/data/liver_cancer')
PATHWAY_FILE = Path(__file__).resolve().parents[2] / 'resources' / 'output_path4.txt'
BIN_COUNT = 8
GRAPH_K = 2
AUC_RANK_THRESHOLD = 0.05
TOP_EDGE_PERCENT = 0.05
FORCE_RECOMPUTE = True

@dataclass(frozen=True)
class TransferDatasetSpec:
    """Describe the train, validation, and query batches for one transfer direction."""
    direction: str
    train_dataset_name: str
    train_batches: List[str]
    valid_batches: List[str]
    test_batches: List[str]

TRANSFER_DATASETS: List[TransferDatasetSpec] = [
    TransferDatasetSpec(
        direction='chc_to_hcc',
        train_dataset_name='chc1_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21',
        train_batches=['1', '2'],
        valid_batches=['3'],
        test_batches=['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
    ),
    TransferDatasetSpec(
        direction='hcc_to_chc',
        train_dataset_name='hcc_all_to_chc',
        train_batches=['4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
        valid_batches=['9'],
        test_batches=['1', '2', '3'],
    ),
]

def dense_float32(matrix_object) -> np.ndarray:
    """Convert an expression matrix to a dense float32 array."""
    if sparse.issparse(matrix_object):
        return matrix_object.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix_object, dtype=np.float32)

def canonical_gene_name(gene_name: str) -> str:
    """Document canonical_gene_name behavior."""
    return str(gene_name).upper().replace(' ', '')

def binned_query_path(batch_id: str) -> Path:
    """Return the binned query AnnData path for one batch."""
    return DATA_DIR / f'liver_cancer_query_binning_{BIN_COUNT}_with_0_{batch_id}.h5ad'

def query_graph_path(batch_id: str) -> Path:
    """Return the query pathway-score and communication-graph path."""
    return DATA_DIR / f'liver_cancer_query_binning_{BIN_COUNT}_with_0_{batch_id}_path_scores_graph_k_{GRAPH_K}.h5ad'

def binned_train_path(dataset_name: str) -> Path:
    """Return the binned training AnnData path."""
    return DATA_DIR / f'liver_cancer_train_binning_{BIN_COUNT}_with_0_{dataset_name}.h5ad'

def train_graph_path(dataset_name: str) -> Path:
    """Return the train-validation communication-graph path."""
    return DATA_DIR / f'liver_cancer_train_binning_{BIN_COUNT}_with_0_{dataset_name}_train_valid_CGraph_k_{GRAPH_K}.h5ad'

def required_query_batches() -> List[str]:
    """Return all query batches required by the transfer experiments."""
    batch_ids = set()
    for spec in TRANSFER_DATASETS:
        batch_ids.update(spec.train_batches)
        batch_ids.update(spec.valid_batches)
        batch_ids.update(spec.test_batches)
    return sorted(batch_ids, key=lambda value: int(value))

def apply_fly_binning_8_with_0(expression_matrix: np.ndarray) -> np.ndarray:
    """Document apply_fly_binning_8_with_0 behavior."""
    binned_matrix = np.empty_like(expression_matrix, dtype=np.float32)
    for row_index in range(expression_matrix.shape[0]):
        row_values = expression_matrix[row_index, :]
        min_value = np.min(row_values)
        max_value = np.max(row_values)
        bin_edges = np.linspace(min_value, max_value, BIN_COUNT + 1)
        binned_matrix[row_index, :] = np.digitize(row_values, bin_edges) - 1
    return binned_matrix

def write_binned_query(batch_id: str) -> Path:
    """Create the binned AnnData file for one query batch."""
    source_path = DATA_DIR / f'liver_cancer_query_{batch_id}.h5ad'
    output_path = binned_query_path(batch_id)
    if output_path.exists() and not FORCE_RECOMPUTE:
        return output_path
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    print(f'[binning-query] batch {batch_id}', flush=True)
    adata = sc.read(source_path)
    adata.X = apply_fly_binning_8_with_0(dense_float32(adata.X))
    adata.write(output_path)
    return output_path

def load_pathway_membership(var_names: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    """Build the gene-by-pathway membership matrix."""
    gene_names = [canonical_gene_name(gene_name) for gene_name in var_names]
    gene_to_index = {gene_name: index for index, gene_name in enumerate(gene_names)}
    pathway_names: List[str] = []
    pathway_gene_indices: List[List[int]] = []
    with PATHWAY_FILE.open('r', encoding='utf-8') as handle:
        for line in handle:
            values = line.strip().split('\t')
            if len(values) < 2:
                continue
            pathway_names.append(values[0])
            matched_gene_indices = sorted({gene_to_index[canonical_gene_name(gene)] for gene in values[1:] if canonical_gene_name(gene) in gene_to_index})
            pathway_gene_indices.append(matched_gene_indices)
    membership_matrix = np.zeros((len(gene_names), len(pathway_names)), dtype=np.uint8)
    for pathway_index, gene_indices in enumerate(pathway_gene_indices):
        membership_matrix[gene_indices, pathway_index] = 1
    return membership_matrix, pathway_names

def calculate_path_scores(adata: ad.AnnData, membership_matrix: np.ndarray, pathway_names: List[str]) -> pd.DataFrame:
    """Calculate AUCell-style pathway scores."""
    expression_matrix = dense_float32(adata.X)
    top_gene_count = max(1, int(expression_matrix.shape[1] * AUC_RANK_THRESHOLD))
    auc_weights = np.ones(top_gene_count, dtype=np.float32)
    if top_gene_count > 1:
        auc_weights[0] = 0.5
        auc_weights[-1] = 0.5
    score_matrix = np.zeros((expression_matrix.shape[0], membership_matrix.shape[1]), dtype=np.float32)
    gene_indices = {canonical_gene_name(gene_name): index for index, gene_name in enumerate(adata.var_names)}
    expression_df = pd.DataFrame(expression_matrix, index=adata.obs_names, columns=[canonical_gene_name(gene_name) for gene_name in adata.var_names])
    for row_position, (_, row_values) in enumerate(expression_df.iterrows()):
        ranked_genes = row_values.sort_values(ascending=False).index.tolist()[:top_gene_count]
        ranked_gene_indices = [gene_indices[gene_name] for gene_name in ranked_genes]
        score_matrix[row_position, :] = auc_weights @ membership_matrix[ranked_gene_indices, :]
        if row_position > 0 and row_position % 1000 == 0:
            print(f'  [path-score] processed {row_position}/{expression_matrix.shape[0]} cells', flush=True)
    path_scores = pd.DataFrame(score_matrix, index=adata.obs_names, columns=pathway_names)
    path_scores = path_scores.loc[:, (path_scores != 0).any(axis=0)]
    return path_scores

def filter_top_similarity_edges(edge_sources: List[int], edge_targets: List[int], similarities: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Document filter_top_similarity_edges behavior."""
    if len(similarities) == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    top_edge_count = int(len(similarities) * TOP_EDGE_PERCENT)
    if top_edge_count == 0:
        top_edge_count = 1
    similarity_array = np.asarray(similarities, dtype=np.float32)
    edge_source_array = np.asarray(edge_sources, dtype=np.int64)
    edge_target_array = np.asarray(edge_targets, dtype=np.int64)
    top_indices = np.argsort(similarity_array)[-top_edge_count:]
    edge_index = np.array([edge_source_array[top_indices], edge_target_array[top_indices]], dtype=np.int64)
    edge_weights = similarity_array[top_indices]
    return edge_index, edge_weights

def build_communication_graph(adata: ad.AnnData, path_scores: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Document build_communication_graph behavior."""
    valid_mask = ~(path_scores == 0).all(axis=1)
    valid_path_scores = path_scores.loc[valid_mask]
    if valid_path_scores.shape[0] < GRAPH_K:
        raise ValueError(f'Not enough valid cells to build graph: {valid_path_scores.shape[0]}')
    valid_cell_names = valid_path_scores.index.tolist()
    cell_name_to_index = {cell_name: index for index, cell_name in enumerate(adata.obs_names)}
    valid_cell_indices = [cell_name_to_index[cell_name] for cell_name in valid_cell_names]
    knn = NearestNeighbors(n_neighbors=GRAPH_K, metric='cosine')
    knn.fit(valid_path_scores.values)
    distances, neighbor_indices = knn.kneighbors(valid_path_scores.values)
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    similarities: List[float] = []
    for local_source_index, neighbors in enumerate(neighbor_indices):
        source_index = valid_cell_indices[local_source_index]
        for neighbor_rank, local_target_index in enumerate(neighbors):
            if local_source_index == local_target_index:
                continue
            target_index = valid_cell_indices[local_target_index]
            similarity = 1 - distances[local_source_index][neighbor_rank]
            edge_sources.append(source_index)
            edge_targets.append(target_index)
            similarities.append(float(similarity))
    edge_index, _ = filter_top_similarity_edges(edge_sources, edge_targets, similarities)
    return {'edge_index': edge_index, 'edge_weights': np.asarray(similarities, dtype=np.float32)}

def write_query_graph(batch_id: str, membership_matrix: np.ndarray, pathway_names: List[str]) -> Path:
    """Write pathway scores and the communication graph for one query batch."""
    output_path = query_graph_path(batch_id)
    if output_path.exists() and not FORCE_RECOMPUTE:
        return output_path
    print(f'[query-graph] batch {batch_id}', flush=True)
    adata = sc.read(binned_query_path(batch_id))
    path_scores = calculate_path_scores(adata, membership_matrix, pathway_names)
    adata.obsm['path_scores'] = path_scores
    adata.uns['communication_graph'] = build_communication_graph(adata, path_scores)
    adata.write(output_path)
    return output_path

def combine_binned_queries(output_path: Path, batch_ids: List[str]) -> Path:
    """Combine binned query batches into a training and validation dataset."""
    if output_path.exists() and not FORCE_RECOMPUTE:
        return output_path
    print(f'[combine-train] {output_path.name}', flush=True)
    adata_list = [sc.read(binned_query_path(batch_id)) for batch_id in batch_ids]
    combined_adata = ad.concat(adata_list, axis=0, join='inner', merge='first', uns_merge='first', index_unique=None)
    combined_adata.uns = dict(adata_list[0].uns)
    combined_adata.write(output_path)
    return output_path

def collect_graph_edges(batch_ids: List[str]) -> np.ndarray:
    """Merge batch-level graph edges with node-index offsets."""
    merged_sources: List[int] = []
    merged_targets: List[int] = []
    node_offset = 0
    for batch_id in batch_ids:
        adata_batch = sc.read(query_graph_path(batch_id))
        edge_index = np.asarray(adata_batch.uns['communication_graph']['edge_index'], dtype=np.int64).copy()
        edge_index[0, :] += node_offset
        edge_index[1, :] += node_offset
        merged_sources.extend(edge_index[0, :].tolist())
        merged_targets.extend(edge_index[1, :].tolist())
        node_offset += adata_batch.shape[0]
    return np.asarray([merged_sources, merged_targets], dtype=np.int64)

def write_train_graph(spec: TransferDatasetSpec) -> Path:
    """Write training and validation graph edges for one transfer direction."""
    train_output_path = train_graph_path(spec.train_dataset_name)
    if train_output_path.exists() and not FORCE_RECOMPUTE:
        return train_output_path
    print(f'[train-graph] {spec.direction}', flush=True)
    train_adata = sc.read(binned_train_path(spec.train_dataset_name))
    train_adata.uns['communication_graph'] = {
        'train_edge_index': collect_graph_edges(spec.train_batches),
        'valid_edge_index': collect_graph_edges(spec.valid_batches),
    }
    train_adata.write(train_output_path)
    return train_output_path

def adata_summary(path: Path) -> Dict[str, object]:
    """Summarize a generated AnnData file for validation and manifests."""
    adata = sc.read(path)
    graph = adata.uns.get('communication_graph', {})
    graph_shapes = {key: list(np.asarray(value).shape) for key, value in graph.items() if key.endswith('edge_index')}
    return {
        'path': str(path),
        'n_obs': int(adata.n_obs),
        'n_vars': int(adata.n_vars),
        'x_min': float(np.min(adata.X)),
        'x_max': float(np.max(adata.X)),
        'batch_counts': {str(key): int(value) for key, value in adata.obs['batch'].astype(str).value_counts().sort_index().items()},
        'graph_shapes': graph_shapes,
    }

def write_manifest(query_outputs: Dict[str, Path], train_outputs: Dict[str, Tuple[Path, Path]]) -> None:
    """Write JSON and CSV manifests for generated datasets."""
    manifest = {
        'binning': 'per-cell np.digitize(row, np.linspace(row.min(), row.max(), 9)) - 1',
        'graph_k': GRAPH_K,
        'auc_rank_threshold': AUC_RANK_THRESHOLD,
        'top_edge_percent': TOP_EDGE_PERCENT,
        'pathway_file': str(PATHWAY_FILE),
        'datasets': {},
        'query_graph_files': {},
    }
    csv_rows: List[Dict[str, object]] = []
    for batch_id, path in query_outputs.items():
        summary = adata_summary(path)
        manifest['query_graph_files'][batch_id] = summary
        csv_rows.append({'role': 'query_graph', 'direction': 'shared', 'batch': batch_id, **summary})
    for direction, (base_train_path, graph_train_path) in train_outputs.items():
        manifest['datasets'][direction] = {
            'base_train': adata_summary(base_train_path),
            'kspada_train': adata_summary(graph_train_path),
        }
        csv_rows.append({'role': 'base_train', 'direction': direction, 'batch': '', **manifest['datasets'][direction]['base_train']})
        csv_rows.append({'role': 'kspada_train', 'direction': direction, 'batch': '', **manifest['datasets'][direction]['kspada_train']})
    json_path = DATA_DIR / f'kspada_transfer_binning_{BIN_COUNT}_with_0_manifest.json'
    csv_path = DATA_DIR / f'kspada_transfer_binning_{BIN_COUNT}_with_0_manifest.csv'
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f'[manifest] {json_path}', flush=True)
    print(f'[manifest] {csv_path}', flush=True)

def main() -> None:
    """Run the complete liver cancer binning and graph preparation workflow."""
    warnings.filterwarnings('ignore', category=UserWarning)
    query_batches = required_query_batches()
    for batch_id in query_batches:
        write_binned_query(batch_id)
    reference_adata = sc.read(binned_query_path(query_batches[0]))
    membership_matrix, pathway_names = load_pathway_membership(reference_adata.var_names)
    query_outputs = {batch_id: write_query_graph(batch_id, membership_matrix, pathway_names) for batch_id in query_batches}
    train_outputs: Dict[str, Tuple[Path, Path]] = {}
    for spec in TRANSFER_DATASETS:
        all_train_valid_batches = spec.train_batches + spec.valid_batches
        base_train_path = combine_binned_queries(binned_train_path(spec.train_dataset_name), all_train_valid_batches)
        graph_train_path = write_train_graph(spec)
        train_outputs[spec.direction] = (base_train_path, graph_train_path)
    write_manifest(query_outputs, train_outputs)
    print('[done] K-SPADA binning_8_with_0 transfer datasets are ready.', flush=True)

if __name__ == '__main__':
    main()
