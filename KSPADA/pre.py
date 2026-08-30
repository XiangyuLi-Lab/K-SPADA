

import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
from .KSPADA_model import scTrans_model as create_model
from .train import compute_edge_weights
from .train import knn_module
from sklearn.preprocessing import LabelEncoder


def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X

def get_weight(att_mat,pathway):
    att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    v = pd.DataFrame(v[0,1:].detach().numpy()).T
    v.columns = pathway
    return v


def mergeCGraph(adata, edge_index):
    graph_comm = adata.uns['communication_graph']['edge_index']

    if isinstance(graph_comm, (list, np.ndarray)):
        graph_comm = torch.tensor(graph_comm, dtype=torch.long)

    if graph_comm.shape[0] != 2:
        raise ValueError("Invalid shape for edge_index while predicting")

    edge_index = torch.cat([edge_index, graph_comm], dim=1)

    return edge_index


def buildPreGraph(adata, batch, label_name='Celltype'):
    """Build the full expression, spatial, and communication graph used for prediction."""
    neighbor = 6
    print('neighbor:', neighbor)

    x = adata.obs['x'].values
    y = adata.obs['y'].values
    coordinates = torch.tensor(list(zip(x, y)), dtype=torch.float32)

    if 'batch' in adata.obs.columns:
        batch_values = adata.obs['batch'].reset_index(drop=True).astype('str')
        batch_pre = LabelEncoder().fit_transform(batch_values)
    else:
        batch_pre = np.zeros(adata.n_obs, dtype=np.int64)
    batch_pre = torch.from_numpy(np.array(batch_pre).astype(np.int64))
    if batch_pre.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "Prediction graph construction failed: batch length must match the number of cells; "
            f"batch={batch_pre.shape[0]}, cells={coordinates.shape[0]}"
        )

    edge_index = knn_module(coordinates, batch_pre, neighbor)
    print('edge_index shape:', edge_index.shape)
    edge_index = mergeCGraph(adata, edge_index)
    print('edge_index after merge shape:', edge_index.shape)

    exp_gcn = todense(adata)
    genes = adata.var_names.values
    exp_gcn = np.array(exp_gcn, dtype=np.float32)
    exp_gcn = torch.from_numpy(exp_gcn)

    label = adata.obs[label_name].astype('str')
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    inverse = label_encoder.inverse_transform(range(0, np.max(label) + 1))
    label = torch.from_numpy(label.astype(np.int64))

    return exp_gcn, label, edge_index, inverse, genes, coordinates


def prediect(adata,model_weight_path,project,mask_path,laten=False,save_att = 'X_att', save_lantent = 'X_lat',n_step=10000,cutoff=0.1,n_unannotated = 0,batch_size = 50,embed_dim=48,depth=2,num_heads=4, use_gcn=False, data_batch=[]):
    """Run full-graph prediction with a trained K-SPADA model."""
    device = torch.device("cpu")
    print(device)

    print("depth:",depth)
    print("num_heads:",num_heads)
    print("embed_dim:",embed_dim)
    print("use_gcn:",use_gcn)
    print("prediction_mode: full_graph")

    num_genes = adata.shape[1]
    mask = np.load(mask_path)
    project_path = os.getcwd()+'/%s'%project
    pathway = pd.read_csv(project_path+'/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path+'/label_dictionary.csv', sep=',',header=0,index_col=0)
    n_c = len(dictionary)
    label_name = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name][i]

    exp_gcn_pre, label_pre, edge_index_pre, inverse, genes, coordinates_pre = buildPreGraph(adata, data_batch, label_name)

    model = create_model(num_classes=n_c, num_genes=num_genes,mask = mask, has_logits=False,depth=depth,num_heads=num_heads, embed_dim=embed_dim, use_gcn=use_gcn).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    parm={}
    for name,parameters in model.named_parameters():
        parm[name]=parameters.detach().cpu().numpy()
    gene2token = parm['feature_embed.fe.weight']
    gene2token = gene2token.reshape((int(gene2token.shape[0]/embed_dim),embed_dim,adata.shape[1]))
    gene2token = abs(gene2token)
    gene2token = np.max(gene2token,axis=1)
    gene2token = pd.DataFrame(gene2token)
    gene2token.columns=adata.var_names
    gene2token.index = pathway['0']
    gene2token.to_csv(project_path+'/gene2token_weights.csv')

    with torch.no_grad():
        exp = exp_gcn_pre.to(device)
        edge_index = edge_index_pre.to(device)
        coordinates = coordinates_pre.to(device)
        edge_weights = compute_edge_weights(coordinates, edge_index)
        lat, pre, _, weights = model(exp, edge_index, edge_weights, coordinates, return_attention=True)

        pre = torch.squeeze(pre).cpu()
        pre = F.softmax(pre, dim=1)
        max_probs, pred_labels = torch.max(pre, dim=1)
        unknown_labels = torch.full_like(pred_labels, fill_value=n_c)
        predict_class = torch.where(max_probs >= cutoff, pred_labels, unknown_labels).numpy()
        pre_class = max_probs.numpy()

        meta = pd.DataFrame({
            'Prediction': predict_class,
            'Probability': pre_class,
        }, index=adata.obs.index.astype('str'))

        if laten:
            latent_np = torch.squeeze(lat).cpu().numpy().astype('float32')
            new = sc.AnnData(latent_np, obs=meta)
        else:
            att = torch.squeeze(weights).cpu().numpy().astype('float32')
            att = att[:,0:(len(pathway)-n_unannotated)]
            varinfo = pd.DataFrame(
                pathway.iloc[0:len(pathway)-n_unannotated,0].values,
                index=pathway.iloc[0:len(pathway)-n_unannotated,0],
                columns=['pathway_index']
            )
            new = sc.AnnData(att, obs=meta, var=varinfo)

    new.obs.index = adata.obs.index
    new.obs['Prediction'] = new.obs['Prediction'].map(dic)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    return(new, lat)


import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Union
import anndata as ad


@contextmanager
def _working_directory(target_directory: Union[str, Path]):
    """Temporarily use the model parent directory for legacy relative paths."""
    resolved_directory = Path(target_directory).expanduser().resolve()
    previous_directory = Path.cwd()
    os.chdir(resolved_directory)
    try:
        yield resolved_directory
    finally:
        os.chdir(previous_directory)


def _read_adata(adata_input, copy: bool = True) -> ad.AnnData:
    """Read an AnnData object or local h5ad path for prediction."""
    if isinstance(adata_input, ad.AnnData):
        return adata_input.copy() if copy else adata_input
    input_path = Path(adata_input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return ad.read_h5ad(input_path)


def latest_model_weight(model_dir: Union[str, Path]) -> Path:
    """Return the checkpoint with the largest model-N epoch number."""
    model_directory = Path(model_dir).expanduser().resolve()
    checkpoint_pattern = re.compile(r"^model-(\d+)\.pth$")
    checkpoint_items = []
    for checkpoint_path in model_directory.glob("model-*.pth"):
        match = checkpoint_pattern.match(checkpoint_path.name)
        if match:
            checkpoint_items.append((int(match.group(1)), checkpoint_path))
    if not checkpoint_items:
        raise FileNotFoundError(f"No model-N.pth checkpoint found in {model_directory}")
    return max(checkpoint_items, key=lambda item: item[0])[1]


def _label_name_from_model(model_dir: Union[str, Path]) -> str:
    """Read the original training label column from label_dictionary.csv."""
    dictionary_path = Path(model_dir).expanduser().resolve() / "label_dictionary.csv"
    if not dictionary_path.is_file():
        raise FileNotFoundError(dictionary_path)
    label_dictionary = pd.read_csv(dictionary_path, index_col=0)
    return str(label_dictionary.columns[0])


def predict_model(
    adata_input,
    model_dir: Union[str, Path],
    model_weight_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    prediction_batches: Optional[Iterable[Union[str, int]]] = None,
    cutoff: float = 0.1,
    use_gcn: bool = True,
    include_match: bool = True,
    **predict_kwargs,
) -> ad.AnnData:
    """Predict cell labels and optionally write the resulting AnnData file."""
    model_directory = Path(model_dir).expanduser().resolve()
    checkpoint_path = Path(model_weight_path).expanduser().resolve() if model_weight_path else latest_model_weight(model_directory)
    query_adata = _read_adata(adata_input, copy=True)
    label_name = _label_name_from_model(model_directory)
    original_has_label = label_name in query_adata.obs
    if not original_has_label:
        query_adata.obs[label_name] = "Unknown"
    prediction_batch_list = [] if prediction_batches is None else [str(value) for value in prediction_batches]
    if use_gcn and not prediction_batch_list:
        if "batch" not in query_adata.obs:
            raise ValueError("prediction_batches or adata.obs['batch'] is required when use_gcn=True")
        prediction_batch_list = sorted(query_adata.obs["batch"].astype(str).unique())
    with _working_directory(model_directory.parent):
        prediction_adata, _latent = prediect(
            query_adata,
            model_weight_path=str(checkpoint_path),
            project=model_directory.name,
            mask_path=str(model_directory / "mask.npy"),
            cutoff=cutoff,
            use_gcn=use_gcn,
            data_batch=prediction_batch_list,
            **predict_kwargs,
        )
    for obs_column in query_adata.obs.columns:
        if obs_column in prediction_adata.obs and len(prediction_adata.obs) == len(query_adata.obs):
            prediction_adata.obs[obs_column] = query_adata.obs[obs_column].to_numpy(copy=True)
    if not original_has_label:
        prediction_adata.obs.drop(columns=[label_name], inplace=True)
    elif include_match:
        prediction_adata.obs["match"] = (
            prediction_adata.obs[label_name].astype(str) == prediction_adata.obs["Prediction"].astype(str)
        ).astype(int)
        prediction_adata.uns["prediction_accuracy"] = float(prediction_adata.obs["match"].mean())
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_adata.write_h5ad(resolved_output_path)
    return prediction_adata


def add_prediction_match(adata_input, label_key: str = "Celltype", prediction_key: str = "Prediction", output_key: str = "match", copy: bool = True) -> ad.AnnData:
    """Add a Boolean correctness column to prediction results."""
    result_adata = _read_adata(adata_input, copy=copy)
    result_adata.obs[output_key] = result_adata.obs[label_key].astype(str) == result_adata.obs[prediction_key].astype(str)
    return result_adata


def prediction_accuracy(adata_input, label_key: str = "Celltype", prediction_key: str = "Prediction") -> float:
    """Return the fraction of correctly predicted cells."""
    result_adata = _read_adata(adata_input, copy=False)
    return float((result_adata.obs[label_key].astype(str) == result_adata.obs[prediction_key].astype(str)).mean())


def prediction_accuracy_by_label(adata_input, label_key: str = "Celltype", prediction_key: str = "Prediction") -> pd.Series:
    """Return prediction accuracy for each true cell label."""
    result_adata = add_prediction_match(adata_input, label_key=label_key, prediction_key=prediction_key, copy=True)
    return result_adata.obs.groupby(label_key, observed=False)["match"].mean().sort_values(ascending=False)
