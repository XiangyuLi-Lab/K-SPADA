import scanpy as sc
import pandas as pd
import numpy as np

from sklearn import metrics

def aucell_manual(expression_matrix, gene_sets, auc_rank_threshold=0.05):
    """Document aucell_manual behavior."""
    all_genes = expression_matrix.columns.str.upper().str.replace(' ', '')
    expression_matrix.columns = all_genes

    auc_scores = pd.DataFrame(index=expression_matrix.index)

    k = int(len(expression_matrix.columns) * auc_rank_threshold)

    for path_name, genes in gene_sets.items():
        selected_genes = [g.upper().replace(' ', '') for g in genes if g.upper().replace(' ', '') in all_genes]
        if len(selected_genes) == 0:
            print(f"[Warning] No valid genes found for path {path_name}")
            auc_scores[path_name] = 0
            continue

        scores = []
        for idx, row in expression_matrix.iterrows():
            ranked_genes = row.sort_values(ascending=False).index.tolist()
            top_k_genes = set(ranked_genes[:k])
            labels = [1 if gene in selected_genes else 0 for gene in ranked_genes[:k]]
            scores.append(metrics.auc(range(len(labels)), labels))

        auc_scores[path_name] = scores

    return auc_scores

def calculatePathEnrichment(adata, path_data):
    """Calculate pathway enrichment scores and store them in the AnnData object."""
    expression_matrix = adata.X
    gene_names = adata.var_names.tolist()
    gene_names = [gene.upper().replace(' ', '') for gene in gene_names]
    expression_matrix = pd.DataFrame(expression_matrix, index=adata.obs_names, columns=gene_names)
    enrichment_scores = {}
    cnt = 0
    for path_name, genes in path_data.items():
        genes = [gene.upper().replace(' ','') for gene in genes]
        selected_genes = [gene for gene in genes if gene in gene_names]
        selected_genes = list(set(selected_genes))
        if len(selected_genes) == 0:
            print(f"Warning: No genes found for path '{path_name}' in the expression matrix.")
            cnt += 1
            enrichment_scores[path_name] = pd.Series(0, index=expression_matrix.index)
            continue
        scores_df = aucell_manual(expression_matrix, {path_name: selected_genes})
        scores = scores_df.loc[:,path_name]
        enrichment_scores[path_name] = scores
    print(f"Warning: {cnt} paths had no genes found in the expression matrix.")
    path_score_matrix = pd.DataFrame(enrichment_scores, index=expression_matrix.index)
    path_score_matrix = path_score_matrix.loc[:, (path_score_matrix != 0).any(axis=0)]
    adata.obsm['path_scores'] = path_score_matrix
    return adata

def getPathScores(adata_name, filename):
    path = "./resources/output_path4.txt"
    path_data = {}
    with open(path, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            path_name = values[0]
            path_data[path_name] = values[1:]

    adata = sc.read(adata_name)
    updated_adata = calculatePathEnrichment(adata, path_data)
    non_zero_elements = np.count_nonzero(adata.obsm['path_scores'])
    print(f"Number of non-zero elements in path_scores: {non_zero_elements}")
    name_ls = adata_name.split('.')
    save_path = filename + '_path_scores.' + name_ls[1]
    updated_adata.write(save_path)
    return updated_adata

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def filter_top_similarity_edges(edge_sources, edge_targets, similarities, top_percent=0.2):
    """Keep edges with similarities in the requested top fraction."""
    num_edges = len(similarities)
    top_k = int(num_edges * top_percent)

    edge_sources = np.asarray(edge_sources)
    edge_targets = np.asarray(edge_targets)
    similarities = np.asarray(similarities)

    if top_k == 0:
        top_k = 1

    top_indices = np.argsort(similarities)[-top_k:]

    filtered_edge_index = np.array([
        edge_sources[top_indices],
        edge_targets[top_indices]
    ], dtype=np.int64)
    filtered_weights = similarities[top_indices]

    return filtered_edge_index, filtered_weights


def getCommunicateGraph(adata_name, k=6):
    updated_adata = sc.read(adata_name)
    path_scores = updated_adata.obsm['path_scores']

    valid_mask = ~(path_scores == 0).all(axis=1)
    valid_path_scores = path_scores.loc[valid_mask]

    valid_cell_names = valid_path_scores.index.tolist()
    cellname_to_index = {cell_name: i for i, cell_name in enumerate(updated_adata.obs_names)}
    valid_cell_ids = [cellname_to_index[cell_name] for cell_name in valid_cell_names]

    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(valid_path_scores.values)
    distances, indices = knn.kneighbors(valid_path_scores.values)

    edge_sources = []
    edge_targets = []
    adata_distances = []
    for i, neighbors in enumerate(indices):
        source_id = valid_cell_ids[i]
        for j, neighbor_idx in enumerate(neighbors):
            if i == neighbor_idx:
                continue
            target_id = valid_cell_ids[neighbor_idx]
            similarity = 1 - distances[i][j]

            edge_sources.append(source_id)
            edge_targets.append(target_id)
            adata_distances.append(similarity)

    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    edge_index, edge_weights = filter_top_similarity_edges(
        edge_sources, edge_targets, adata_distances, top_percent=0.2
    )

    updated_adata.uns['communication_graph'] = {'edge_index': edge_index}
    updated_adata.uns['communication_graph']['edge_weights'] = adata_distances

    print(f"Graph edge_index shape: {edge_index.shape} with {edge_index.shape[1]} edges.")

    return updated_adata


def CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name, file_name,k=6 ):
    communicate_graph_all = {'train_edge_index': [[], []], 'valid_edge_index': [[], []]}
    node_offset  = 0
    for batch in train_batch:
        adata_name = file_pre + batch + '_path_scores_graph_k_' + str(k) + '.h5ad'
        adata_batch = sc.read(adata_name)
        edge_index_batch = adata_batch.uns['communication_graph']['edge_index']
        edge_weights_batch = adata_batch.uns['communication_graph']['edge_weights']
        edge_index_batch[0, :] += node_offset
        edge_index_batch[1, :] += node_offset
        node_offset  += adata_batch.shape[0]
        communicate_graph_all['train_edge_index'][0].extend(edge_index_batch[0].tolist())
        communicate_graph_all['train_edge_index'][1].extend(edge_index_batch[1].tolist())

    node_offset  = 0
    for batch in valid_batch:
        adata_name = file_pre + batch + '_path_scores_graph_k_' + str(k) + '.h5ad'
        adata_batch = sc.read(adata_name)
        edge_index_batch = adata_batch.uns['communication_graph']['edge_index']
        edge_weights_batch = adata_batch.uns['communication_graph']['edge_weights']
        edge_index_batch[0, :] += node_offset
        edge_index_batch[1, :] += node_offset
        node_offset  += adata_batch.shape[0]
        communicate_graph_all['valid_edge_index'][0].extend(edge_index_batch[0].tolist())
        communicate_graph_all['valid_edge_index'][1].extend(edge_index_batch[1].tolist())

    train_adata = sc.read(train_adata_name)
    train_adata.uns['communication_graph'] = communicate_graph_all
    train_adata_name_ls = train_adata_name.split('.')
    new_adata_name = file_name + f'_train_valid_CGraph_k_{k}.' + train_adata_name_ls[1]
    train_adata.write(new_adata_name)


if __name__ == "__main__":


    batch_ls = ['1', '2', '3',
                '4', '5', '6',
                '7', '8', '9', '10',
                '11', '12', '13',
                '14', '15', '16',
                '17', '18', '19', '20',
                '21']
    file_pre = '/home/lixiangyu/TOSICA-main/data/liver_cancer/liver_cancer_query_'
    k = 2


    train_adata_file_pre = '/home/lixiangyu/TOSICA-main/data/liver_cancer/liver_cancer_train_'
    train_adata_name_ls = ['chc1_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21.h5ad',
                            'hcc2_1_2_3_21.h5ad']
    train_adata_name_ls = [
        '1_2_3_4_to_5',
        '1_2_3_5_to_4',
        '1_2_4_5_to_3',
        '1_3_4_5_to_2',
        '2_3_4_5_to_1'
    ]
    train_adata_name_ls = [
        'hcc_all_to_chc'
    ]
    train_adata_name_ls = [
        'hcc1_to_chc',
        'hcc3_to_chc',
        'hcc4_to_chc',
        'hcc5_to_chc'
    ]
    train_adata_name_ls = [
        'chc_L', 'chc_N', 'chc_T', 'hcc_L', 'hcc_N', 'hcc_T'
    ]
    train_batch_ls = [['1', '2'],
                      ['7', '8', '9']]
    train_batch_ls = [
        ['4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16'],
        ['4', '5', '6', '7', '8', '10', '11', '12', '13', '17', '18', '19', '20'],
        ['4', '5', '6', '7', '8', '10', '14', '15', '16', '17', '18', '19', '20'],
        ['4', '5', '6', '11', '13', '14', '15', '16', '17', '18', '19', '20'],
        ['7', '8', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    ]
    train_batch_ls = [
        ['4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    ]
    train_batch_ls = [
        ['4', '5', '6'],
        ['11', '12', '13'],
        ['14', '15', '16'],
        ['17', '18', '19', '20']
    ]
    train_batch_ls = [
        ['1'],
        ['2'],
        ['3'],
        ['4', '7', '11', '14', '17'],
        ['5', '8', '12', '15', '18'],
        ['6', '9', '13', '16', '19']
    ]
    valid_batch_ls = [
        ['9'],
                     ['9'],
                      ['9'],
                      ['12'],
                      ['9']
                      ]
    valid_batch_ls = [
        ['9']
    ]
    valid_batch_ls = [
        ['6'],
        ['13'],
        ['16'],
        ['20']
    ]
    valid_batch_ls = [
        ['4'],
        ['5'],
        ['6'],
        ['17'],
        ['18'],
        ['19']
    ]
    for i, train_adata_name in enumerate(train_adata_name_ls):
        train_batch = train_batch_ls[i]
        valid_batch = valid_batch_ls[i]
        train_adata_name_new = train_adata_file_pre + train_adata_name + '.h5ad'
        file_name = '/home/lixiangyu/sh/program/data/DLPFC/' + train_adata_name
        CreateTrainData(train_batch, valid_batch, file_pre, train_adata_name_new, file_name,k=k)


    print('done!!')


from pathlib import Path
from typing import Iterable, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


AnnDataLike = Union[str, Path, ad.AnnData]


def read_adata(adata_input: AnnDataLike, copy: bool = True) -> ad.AnnData:
    """Read an AnnData object from memory or a .h5ad file.

    Args:
        adata_input: AnnData object or .h5ad path.
        copy: Whether to copy an in-memory AnnData object.

    Returns:
        AnnData: Loaded AnnData.

    Raises:
        TypeError: If the input type is unsupported.
    """

    if isinstance(adata_input, ad.AnnData):
        return adata_input.copy() if copy else adata_input
    input_path = Path(adata_input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return ad.read_h5ad(input_path)


def default_pathway_file() -> Path:
    """Return the packaged pathway file used by the CGraph path score step.

    Returns:
        Path: KSPADA/resources/output_path4.txt.
    """
    return Path(__file__).resolve().parent / "resources" / "output_path4.txt"


def read_pathway_data(pathway_file: Optional[Union[str, Path]] = None) -> dict:
    """Read a tab-delimited pathway file into a pathway-to-genes dictionary.

    Args:
        pathway_file: Optional pathway file. Defaults to packaged output_path4.txt.

    Returns:
        dict: Mapping from pathway name to gene list.

    Raises:
        FileNotFoundError: If the pathway file is missing.
    """
    resolved_pathway_file = Path(pathway_file).expanduser().resolve() if pathway_file else default_pathway_file()
    if not resolved_pathway_file.exists():
        raise FileNotFoundError(f"Pathway file does not exist: {resolved_pathway_file}")
    pathway_data = {}
    with resolved_pathway_file.open("r") as pathway_handle:
        for line in pathway_handle:
            values = line.strip().split("\t")
            if len(values) >= 2:
                pathway_data[values[0]] = values[1:]
    return pathway_data


def densify_for_path_scores(adata_input: ad.AnnData) -> ad.AnnData:
    """Return a copy whose X matrix is dense because the legacy AUCell code expects dense X.

    Args:
        adata_input: AnnData with sparse or dense X.

    Returns:
        AnnData: Copy with dense X.
    """
    dense_adata = adata_input.copy()
    if sparse.issparse(dense_adata.X):
        dense_adata.X = dense_adata.X.toarray()
    return dense_adata


def add_path_scores(
    adata_input: AnnDataLike,
    pathway_file: Optional[Union[str, Path]] = None,
    copy: bool = True,
) -> ad.AnnData:
    """Calculate pathway enrichment scores and store them in adata.obsm['path_scores'].

    Args:
        adata_input: AnnData object or .h5ad path.
        pathway_file: Optional pathway file. Defaults to packaged output_path4.txt.
        copy: Whether to return a new object.

    Returns:
        AnnData: AnnData with obsm['path_scores'].
    """
    adata_object = read_adata(adata_input, copy=copy)
    pathway_data = read_pathway_data(pathway_file)
    path_score_adata = densify_for_path_scores(adata_object)
    return calculatePathEnrichment(path_score_adata, pathway_data)


def build_communication_graph(
    adata_input: AnnDataLike,
    k: int = 6,
    top_percent: float = 0.05,
    copy: bool = True,
) -> ad.AnnData:
    """Build a communication graph from adata.obsm['path_scores'].

    Args:
        adata_input: AnnData object or .h5ad path with obsm['path_scores'].
        k: Number of nearest neighbors used before edge filtering.
        top_percent: Fraction of strongest cosine-similarity edges to keep.
        copy: Whether to return a new object.

    Returns:
        AnnData: AnnData with uns['communication_graph'].

    Raises:
        ValueError: If path scores are missing or k/top_percent are invalid.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    graph_adata = read_adata(adata_input, copy=copy)
    if "path_scores" not in graph_adata.obsm:
        raise ValueError("Missing adata.obsm['path_scores']; run add_path_scores first")
    path_scores = graph_adata.obsm["path_scores"]
    if not isinstance(path_scores, pd.DataFrame):
        path_scores = pd.DataFrame(path_scores, index=graph_adata.obs_names)
    valid_mask = ~(path_scores == 0).all(axis=1)
    valid_path_scores = path_scores.loc[valid_mask]
    if valid_path_scores.shape[0] <= 1:
        graph_adata.uns["communication_graph"] = {
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weights": np.empty((0,), dtype=np.float32),
        }
        return graph_adata
    neighbor_count = min(k, valid_path_scores.shape[0])
    nearest_neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
    nearest_neighbors.fit(valid_path_scores.values)
    distances, indices = nearest_neighbors.kneighbors(valid_path_scores.values)
    valid_cell_names = valid_path_scores.index.tolist()
    cell_name_to_index = {cell_name: index for index, cell_name in enumerate(graph_adata.obs_names)}
    valid_cell_indices = [cell_name_to_index[cell_name] for cell_name in valid_cell_names]
    edge_sources = []
    edge_targets = []
    edge_similarities = []
    for source_position, neighbor_positions in enumerate(indices):
        source_index = valid_cell_indices[source_position]
        for neighbor_order, neighbor_position in enumerate(neighbor_positions):
            if source_position == neighbor_position:
                continue
            target_index = valid_cell_indices[neighbor_position]
            edge_sources.append(source_index)
            edge_targets.append(target_index)
            edge_similarities.append(1 - distances[source_position][neighbor_order])
    if not edge_sources:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weights = np.empty((0,), dtype=np.float32)
    else:
        edge_index, edge_weights = filter_top_similarity_edges(
            np.asarray(edge_sources),
            np.asarray(edge_targets),
            np.asarray(edge_similarities),
            top_percent=top_percent,
        )
    graph_adata.uns["communication_graph"] = {"edge_index": edge_index, "edge_weights": np.asarray(edge_weights)}
    return graph_adata


def prepare_query_cgraph(
    input_h5ad: Union[str, Path],
    output_h5ad: Union[str, Path],
    pathway_file: Optional[Union[str, Path]] = None,
    k: int = 6,
    top_percent: float = 0.05,
    force: bool = False,
) -> Path:
    """Create a query h5ad with path_scores and communication_graph.

    Args:
        input_h5ad: Input .h5ad path.
        output_h5ad: Output .h5ad path.
        pathway_file: Optional pathway file for path scoring.
        k: Number of nearest neighbors.
        top_percent: Fraction of strongest edges retained.
        force: Overwrite output if it already exists.

    Returns:
        Path: Output .h5ad path.

    Raises:
        FileExistsError: If output exists and force is False.
    """
    output_path = Path(output_h5ad).expanduser().resolve()
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_score_adata = add_path_scores(input_h5ad, pathway_file=pathway_file)
    graph_adata = build_communication_graph(path_score_adata, k=k, top_percent=top_percent, copy=False)
    graph_adata.write_h5ad(output_path)
    return output_path


def create_train_valid_cgraph(
    train_batches: Iterable[Union[str, int]],
    valid_batches: Iterable[Union[str, int]],
    batch_file_prefix: Union[str, Path],
    train_adata_path: Union[str, Path],
    output_prefix: Union[str, Path],
    k: int = 6,
    force: bool = False,
) -> Path:
    """Merge per-batch communication graphs into one train-valid CGraph h5ad.

    Args:
        train_batches: Batch IDs used for training.
        valid_batches: Batch IDs used for validation.
        batch_file_prefix: Prefix before each batch ID, matching legacy CreateTrainData contract.
        train_adata_path: Full training AnnData path.
        output_prefix: Output prefix before _train_valid_CGraph_k_{k}.h5ad.
        k: Communication graph k used in per-batch files.
        force: Overwrite output if it already exists.

    Returns:
        Path: Final train-valid CGraph h5ad path.
    """
    train_batch_list = [str(batch_id) for batch_id in train_batches]
    valid_batch_list = [str(batch_id) for batch_id in valid_batches]
    output_base = Path(output_prefix).expanduser().resolve()
    output_path = output_base.parent / f"{output_base.name}_train_valid_CGraph_k_{k}.h5ad"
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CreateTrainData(
        train_batch=train_batch_list,
        valid_batch=valid_batch_list,
        file_pre=str(Path(batch_file_prefix).expanduser()),
        train_adata_name=str(Path(train_adata_path).expanduser()),
        file_name=str(output_base),
        k=k,
    )
    return output_path


def validate_cgraph_fields(adata_input: AnnDataLike, mode: str = "query") -> bool:
    """Validate the CGraph fields required by train or prediction.

    Args:
        adata_input: AnnData object or .h5ad path.
        mode: query requires edge_index; train requires train_edge_index and valid_edge_index.

    Returns:
        bool: True when required fields exist.

    Raises:
        ValueError: If required fields are missing.
    """
    adata_object = read_adata(adata_input, copy=False)
    communication_graph = adata_object.uns.get("communication_graph")
    if communication_graph is None:
        raise ValueError("Missing adata.uns['communication_graph']")
    if mode == "query":
        required_keys = ["edge_index"]
    elif mode == "train":
        required_keys = ["train_edge_index", "valid_edge_index"]
    else:
        raise ValueError("mode must be 'query' or 'train'")
    missing_keys = [key for key in required_keys if key not in communication_graph]
    if missing_keys:
        raise ValueError(f"Missing communication_graph keys: {missing_keys}")
    return True
