import random
import numpy as np

from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict

import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import time
import platform

from .KSPADA_model import scTrans_model as create_model
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import knn_graph
from imblearn.over_sampling import SMOTE

import torch.nn as nn

GENE_TOKEN_GMT_NAME = 'gene_token'


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def todense(adata):
    import scipy
    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse.csc_matrix):
        return adata.X.todense()
    else:
        return adata.X


def knn_module(x, batch, k):
    edge_index = []
    if len(batch) == 1:
        return knn_graph(x, k, loop=False)

    sum = 0
    for b in torch.unique(batch):
        batch_indices = (batch == b)
        batch_points = x[batch_indices]
        if torch.sum(batch_indices) < k:
            continue
        batch_edge_index = knn_graph(batch_points, k, loop=False)
        batch_edge_index += sum
        sum += batch_points.shape[0]
        edge_index.append(batch_edge_index)
    if len(edge_index) > 0:
        edge_index = torch.cat(edge_index, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index


class MyDataSet(Dataset):
    """
    Preproces input matrix and labels.

    """
    def __init__(self, exp, label, edge_index, edge_weight, edges=5):
        self.exp = exp
        self.label = label
        self.len = len(label)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.edges = edges

    def compute_graph(self, node_index):
        edge_indices = (self.edge_index[0] == node_index) | (self.edge_index[1] == node_index)
        sub_edge_index = self.edge_index[:, edge_indices]
        sub_edge_weight = self.edge_weight[edge_indices]

        while sub_edge_index.shape[1] < self.edges:
            sub_edge_index = torch.cat((sub_edge_index, sub_edge_index), dim=1)
            sub_edge_weight = torch.cat((sub_edge_weight, sub_edge_weight))

        if sub_edge_index.shape[1] > self.edges:
            sub_edge_index = sub_edge_index[:, :self.edges]
            sub_edge_weight = sub_edge_weight[:self.edges]

        return sub_edge_index, sub_edge_weight

    def __getitem__(self,index):
        return self.exp[index], self.label[index], self.edge_index, self.edge_weight

    def __len__(self):
        return self.len

def balance_populations(data):
    ct_names = np.unique(data[:,-1])
    ct_counts = pd.value_counts(data[:,-1])
    max_val = min(ct_counts.max(),np.int32(2000000/len(ct_counts)))
    balanced_data = np.empty(shape=(1,data.shape[1]),dtype=np.float32)
    for ct in ct_names:
        tmp = data[data[:,-1] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        tmp_X = tmp[idx]
        balanced_data = np.r_[balanced_data,tmp_X]
    return np.delete(balanced_data,0,axis=0)

def splitDataSet(adata, edge_index, edge_weights, label_name='Celltype', tr_ratio= 0.7):
    """
    Split data set into training set and test set.

    """
    label_encoder = LabelEncoder()
    el_data = pd.DataFrame(todense(adata),index=np.array(adata.obs_names).tolist(), columns=np.array(adata.var_names).tolist())
    el_data[label_name] = adata.obs[label_name].astype('str')
    genes = el_data.columns.values[:-1]
    el_data = np.array(el_data)
    el_data[:,-1] = label_encoder.fit_transform(el_data[:,-1])
    inverse = label_encoder.inverse_transform(range(0,np.max(el_data[:,-1])+1))
    el_data = el_data.astype(np.float32)
    n_genes = len(el_data[1])-1
    train_size = int(len(el_data) * tr_ratio)
    train_dataset, valid_dataset = torch.utils.data.random_split(el_data, [train_size,len(el_data)-train_size])
    exp_train = torch.from_numpy(np.array(train_dataset)[:,:n_genes].astype(np.float32))
    label_train = torch.from_numpy(np.array(train_dataset)[:,-1].astype(np.int64))
    exp_valid = torch.from_numpy(np.array(valid_dataset)[:,:n_genes].astype(np.float32))
    label_valid = torch.from_numpy(np.array(valid_dataset)[:,-1].astype(np.int64))

    train_indices = torch.tensor(train_dataset.indices)
    valid_indices = torch.tensor(valid_dataset.indices)

    mask = torch.any(torch.eq(edge_index[0].view(-1, 1), train_indices), dim=1) | torch.any(
        torch.eq(edge_index[1].view(-1, 1), train_indices), dim=1)

    edge_index_train = edge_index[:, mask]
    edge_weights_train = edge_weights[mask]

    mask_valid = torch.any(torch.eq(edge_index[0].view(-1, 1), valid_indices), dim=1) | torch.any(
        torch.eq(edge_index[1].view(-1, 1), valid_indices), dim=1)
    edge_index_valid = edge_index[:, mask_valid]
    edge_weights_valid = edge_weights[mask_valid]

    return exp_train, label_train, exp_valid, label_valid, edge_index_train, edge_weights_train, edge_index_valid, edge_weights_valid, inverse, genes

def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"],
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"],
        "mouse_reactome": [root / "resources/m_reactome.gmt"],
        "mouse_tf": [root / "resources/m_TF.gmt"],
        "fly": [root / "resources/GO_Biological_Process_2018.gmt"]
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train the model and updata weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, label = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()
        loss = loss_function(pred, label.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        exp, labels = data
        sample_num += exp.shape[0]
        _,pred,_ = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def compute_edge_weights(x, edge_index):
    point1 = x[edge_index[0]]
    point2 = x[edge_index[1]]
    edge_weights = torch.sqrt(torch.sum((point1 - point2) ** 2, dim=1))
    return edge_weights

def test():
    exp = torch.randn(10, 5)
    label = torch.randint(0, 2, (10,))
    edge_index = torch.randint(0, 10, (2, 20))
    edge_weight = torch.randn(20)

    dataset = MyDataSet(exp, label, edge_index, edge_weight)
    print(exp)
    print(edge_index)
    for i in range(len(dataset)):
        exp_i, label_i, edge_index_i, edge_weight_i = dataset[i]
        print(f"Sample {i}:")
        print(f"Expression: {exp_i}")
        print(f"Label: {label_i}")
        print(f"Edge index: {edge_index_i}")
        print(f"Edge weight: {edge_weight_i}")
        print()

def duplicate_nodes(edge_index, node_idx, nodes_to_duplicate):
    new_node_idx = torch.arange(node_idx.max() + 1, node_idx.max() + 1 + len(nodes_to_duplicate))

    new_edges = []
    for old_node, new_node in zip(nodes_to_duplicate, new_node_idx):
        neighbors = edge_index[1][edge_index[0] == old_node]
        new_edges.extend([(new_node, neighbor) for neighbor in neighbors])

    edge_index = torch.cat([edge_index, torch.tensor(new_edges).t().contiguous()], dim=1)

    node_idx = torch.cat([node_idx, new_node_idx])

    return edge_index, node_idx


def get_nodes_to_duplicate(batches_celltype):
    nodes_to_duplicate_all = []
    sum = 0
    for i in range(len(batches_celltype)):
        label_counts = {}
        label_counts = batches_celltype[i].value_counts()

        max_count = label_counts.max()

        nodes_to_duplicate = []
        batch = pd.DataFrame(batches_celltype[i], columns=['Celltype'])

        label_indices = batch.groupby('Celltype').indices

        for label, count in label_counts.items():
            if count == max_count or count == 0:
                continue
            num_to_duplicate = max_count - count
            nodes = label_indices[label]
            if num_to_duplicate <= len(nodes):
                nodes = nodes[:num_to_duplicate]
            else:
                temp = nodes
                while num_to_duplicate > len(nodes):
                    nodes = np.concatenate((nodes, temp), axis=0)
                nodes = nodes[:num_to_duplicate]
            nodes += sum
            nodes_to_duplicate.extend(nodes)
        nodes_to_duplicate_all.extend(nodes_to_duplicate)
        sum += len(batches_celltype[i])
    return nodes_to_duplicate_all


def oversample_graph(adata, edge_index_train, node_idx_train, edge_index_valid, node_idx_valid,
                     coordinates_train, coordinates_valid, train_batch):
    all_batch = adata.obs['batch'].unique().tolist()
    batches_train_celltype = []
    for i in range(len(train_batch)):
        batch_celltype = adata.obs.loc[adata.obs['batch'] == train_batch[i], 'Celltype']
        batches_train_celltype.append(batch_celltype)

    valid_batch = list(set(all_batch).difference(set(train_batch)))
    valid_batch.sort()
    batches_valid_celltype = []
    for i in range(len(valid_batch)):
        batch_celltype = adata.obs.loc[adata.obs['batch'] == valid_batch[i], 'Celltype']
        batches_valid_celltype.append(batch_celltype)

    nodes_to_duplicate_train = get_nodes_to_duplicate(batches_train_celltype)
    nodes_to_duplicate_valid = get_nodes_to_duplicate(batches_valid_celltype)
    if len(nodes_to_duplicate_train) == 0:
        new_edge_index_train = edge_index_train
        new_node_idx_train = node_idx_train
    else:
        new_edge_index_train, new_node_idx_train = duplicate_nodes(edge_index_train, node_idx_train, nodes_to_duplicate_train)
    if len(nodes_to_duplicate_valid) == 0:
        new_edge_index_valid = edge_index_valid
        new_node_idx_valid = node_idx_valid
    else:
        new_edge_index_valid, new_node_idx_valid = duplicate_nodes(edge_index_valid, node_idx_valid, nodes_to_duplicate_valid)


    coordinates_train = torch.cat([coordinates_train, coordinates_train[nodes_to_duplicate_train]], dim=0)
    coordinates_valid = torch.cat([coordinates_valid, coordinates_valid[nodes_to_duplicate_valid]], dim=0)

    return new_edge_index_train, new_node_idx_train, new_edge_index_valid, new_node_idx_valid, coordinates_train, coordinates_valid, nodes_to_duplicate_train, nodes_to_duplicate_valid


def enhanceGraph(edge_index_train, edge_index_valid, coordinates_train, coordinates_valid, batch_train, batch_valid):
    distance_train = compute_edge_weights(coordinates_train, edge_index_train)
    distance_valid = compute_edge_weights(coordinates_valid, edge_index_valid)
    print(edge_index_train.shape)
    print(edge_index_valid.shape)
    print('enhance: ', edge_index_train.shape)
    print('enhance: ', edge_index_valid.shape)
    return edge_index_train, edge_index_valid


def mergeGraph(adata, edge_index_train, edge_index_valid):
    graph_comm_train = adata.uns['communication_graph']['train_edge_index']
    graph_comm_valid = adata.uns['communication_graph']['valid_edge_index']

    if isinstance(graph_comm_train, (list, np.ndarray)):
        graph_comm_train = torch.tensor(graph_comm_train, dtype=torch.long)
    if isinstance(graph_comm_valid, (list, np.ndarray)):
        graph_comm_valid = torch.tensor(graph_comm_valid, dtype=torch.long)

    if graph_comm_train.shape[0] != 2:
        raise ValueError("Invalid shape for train_edge_index")
    if graph_comm_valid.shape[0] != 2:
        raise ValueError("Invalid shape for valid_edge_index")

    edge_index_train = torch.cat([edge_index_train, graph_comm_train], dim=1)
    edge_index_valid = torch.cat([edge_index_valid, graph_comm_valid], dim=1)

    return edge_index_train, edge_index_valid

def buildgraph(adata, train_batch, label_name='Celltype', batch_size=64, neighbor=6):
    mask = adata.obs['batch'].isin(train_batch)
    x_train = adata.obs.loc[mask, 'x']
    y_train = adata.obs.loc[mask, 'y']
    x_valid = adata.obs.loc[~mask, 'x']
    y_valid = adata.obs.loc[~mask, 'y']
    coordinates_train = torch.tensor(list(zip(x_train, y_train)))
    coordinates_valid = torch.tensor(list(zip(x_valid, y_valid)))
    batch_train = adata.obs.loc[mask, 'batch']
    batch_valid = adata.obs.loc[~mask, 'batch']

    batch_train = torch.from_numpy(np.array(batch_train).astype(np.int64))
    batch_valid = torch.from_numpy(np.array(batch_valid).astype(np.int64))
    edge_index_train = knn_module(coordinates_train, batch_train, neighbor)
    edge_index_valid = knn_module(coordinates_valid, batch_valid, neighbor)


    edge_index_train, edge_index_valid = enhanceGraph(edge_index_train, edge_index_valid, coordinates_train, coordinates_valid, batch_train, batch_valid)


    node_idx_train = torch.arange(coordinates_train.shape[0])
    node_idx_valid = torch.arange(coordinates_valid.shape[0])

    edge_index_train, node_idx_train, edge_index_valid, node_idx_valid, coordinates_train, coordinates_valid, nodes_to_duplicate_train, nodes_to_duplicate_valid = oversample_graph(adata, edge_index_train, node_idx_train, edge_index_valid, node_idx_valid, coordinates_train, coordinates_valid, train_batch)

    print("edge_index_train shape before merge:", edge_index_train.shape)
    print("edge_index_valid shape before merge:", edge_index_valid.shape)
    edge_index_train, edge_index_valid = mergeGraph(adata, edge_index_train, edge_index_valid)
    print("edge_index_train shape after merge:", edge_index_train.shape)
    print("edge_index_valid shape after merge:", edge_index_valid.shape)

    sampler_train = NeighborSampler(
        edge_index_train, node_idx=node_idx_train,
        sizes=[neighbor],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    sampler_valid = NeighborSampler(
        edge_index_valid, node_idx=node_idx_valid,
        sizes=[neighbor],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    exp_gcn = pd.DataFrame(todense(adata), index=np.array(adata.obs_names).tolist(),
                           columns=np.array(adata.var_names).tolist())
    genes = exp_gcn.columns.values
    exp_gcn = np.array(exp_gcn)
    exp_gcn = exp_gcn.astype(np.float32)
    exp_gcn_train = exp_gcn[mask]
    exp_gcn_valid = exp_gcn[~mask]

    exp_gcn_train = np.concatenate((exp_gcn_train, exp_gcn_train[nodes_to_duplicate_train]), axis=0)
    exp_gcn_valid = np.concatenate((exp_gcn_valid, exp_gcn_valid[nodes_to_duplicate_valid]), axis=0)

    exp_gcn_train = torch.from_numpy(np.array(exp_gcn_train).astype(np.float32))
    exp_gcn_valid = torch.from_numpy(np.array(exp_gcn_valid).astype(np.float32))

    label = adata.obs[label_name].astype('str')
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    inverse = label_encoder.inverse_transform(range(0, np.max(label) + 1))
    label = label.astype(np.float32)
    label_train = label[mask]
    label_valid = label[~mask]

    label_train = np.concatenate((label_train, label_train[nodes_to_duplicate_train]), axis=0)
    label_valid = np.concatenate((label_valid, label_valid[nodes_to_duplicate_valid]), axis=0)

    label_train = torch.from_numpy(np.array(label_train).astype(np.int64))
    label_valid = torch.from_numpy(np.array(label_valid).astype(np.int64))
    return sampler_train, sampler_valid, exp_gcn_train, exp_gcn_valid, label_train, label_valid, edge_index_train, edge_index_valid, inverse, genes, coordinates_train, coordinates_valid


class GraphLaplacianLoss(nn.Module):
    def __init__(self, lambda_reg):
        super(GraphLaplacianLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, pred, labels, laplacian_matrix):
        ce_loss = self.cross_entropy_loss(pred, labels)

        laplacian_reg = torch.trace(torch.matmul(torch.matmul(embeddings.t(), laplacian_matrix), embeddings))

        total_loss = ce_loss + self.lambda_reg * laplacian_reg

        return total_loss


def compute_laplacian_matrix(edge_index, num_nodes):
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1

    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
    diag = torch.diag(degree_matrix)
    degree_matrix = degree_matrix + torch.diag((diag == 0).float() * 1e-6)

    degree_matrix_sqrt_inv = torch.sqrt(torch.inverse(degree_matrix))
    laplacian_matrix = torch.eye(adj_matrix.shape[0]) - torch.matmul(torch.matmul(degree_matrix_sqrt_inv, adj_matrix), degree_matrix_sqrt_inv)

    return laplacian_matrix


def fit_model(adata, gmt_path, project = None, pre_weights='', label_name='Celltype',max_g=300,max_gs=300, mask_ratio = 0.015,n_unannotated = 1,batch_size=4, embed_dim=48,depth=2,num_heads=4,lr=0.001, epochs= 10, lrf=0.01, use_gcn=False, data_batch = [], **kwargs):
    random_seed = kwargs.get('random_seed', 1)
    weight_decay = kwargs.get('weight_decay', 5E-5)
    set_seed(random_seed)

    lambda_reg = 0.000001
    batch_size_gcn = 64
    neighbor = 6

    for key, value in kwargs.items():
        if key == 'batch_size_gcn':
            batch_size_gcn = value
        if key == 'lambda_reg':
            lambda_reg = value
        if key == 'neighbor':
            neighbor = value

    print('batch_size_gcn:', batch_size_gcn)
    print('lambda_reg:', lambda_reg)
    print('neighbor:', neighbor)
    print('num_head:', num_heads)
    print('depth:', depth)
    print('embed_dim:', embed_dim)
    print('weight_decay:', weight_decay)
    print('random_seed:', random_seed)

    device = 'cuda:0'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)

    sampler_train, sampler_valid, exp_gcn_train, exp_gcn_valid, label_train, label_valid, edge_index_train, edge_index_valid, inverse, genes, coordinates_train, coordinates_valid = buildgraph(adata, data_batch, label_name, batch_size=batch_size_gcn, neighbor=neighbor)
    tb_writer = SummaryWriter()
    if gmt_path == GENE_TOKEN_GMT_NAME:
        mask = np.eye(len(genes), dtype=np.float32)
        pathway = np.array(genes, dtype=str)
        print('Gene token mask loaded!')
    elif gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=0, max_g=max_g)
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=reactome_dict,
                                          add_missing=n_unannotated,
                                          fully_connected=True)
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):])]
        print('Mask loaded!')
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv')
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = len(inverse)
    print('use_gcn:{}'.format(use_gcn))
    model = create_model(num_classes=num_classes, num_genes=len(exp_gcn_train[0]),  mask = mask, embed_dim=embed_dim,depth=depth,num_heads=num_heads,has_logits=False, use_gcn=use_gcn).to(device)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]


    optimizer = optim.Adam(pg, lr=lr, weight_decay=weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]

    loss_function = torch.nn.CrossEntropyLoss()


    for epoch in range(epochs):
        model.train()

        save_attention_this_epoch = (epoch == epochs - 1)

        attention_save_dir = project_path + "/train_attention"
        os.makedirs(attention_save_dir, exist_ok=True)

        accu_loss = torch.zeros(1).to(device)
        accu_num = torch.zeros(1).to(device)
        optimizer.zero_grad()
        sample_num = 0

        data_loader = tqdm(sampler_train)

        for step, data in enumerate(data_loader):
            _, n_id, adjs = data

            exp = exp_gcn_train[n_id].to(device)
            edge_index, _, _ = adjs.to(device)

            coordinates_input = coordinates_train[n_id].to(device)
            edge_weights = compute_edge_weights(coordinates_input, edge_index)

            sample_num += exp.shape[0]

            label_temp = label_train[n_id].to(device)

            return_attention = save_attention_this_epoch

            if return_attention:
                _, pred, _, attn_weights = model(
                    exp.to(device),
                    edge_index,
                    edge_weights,
                    coordinates_input,
                    return_attention=True
                )
            else:
                _, pred, _ = model(
                    exp.to(device),
                    edge_index,
                    edge_weights,
                    coordinates_input,
                    return_attention=False
                )
                attn_weights = None

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, label_temp).sum().item()

            loss = loss_function(pred, label_temp)
            loss.backward()

            accu_loss += loss.item()

            data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num
            )

            optimizer.step()
            optimizer.zero_grad()

            if save_attention_this_epoch and attn_weights is not None:
                attn_np = attn_weights.detach().cpu().to(torch.float16).numpy()
                node_np = n_id.detach().cpu().numpy()

                np.save(
                    attention_save_dir + "/epoch_{}_step_{}_attention.npy".format(epoch, step),
                    attn_np
                )

                np.save(
                    attention_save_dir + "/epoch_{}_step_{}_node_id.npy".format(epoch, step),
                    node_np
                )

            del exp, label_temp, pred, loss, data, edge_index, edge_weights, coordinates_input

            if attn_weights is not None:
                del attn_weights

            train_loss, train_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
            torch.cuda.empty_cache()


        model.eval()
        with torch.no_grad():
            accu_num = torch.zeros(1).to(device)
            accu_loss = torch.zeros(1).to(device)
            sample_num = 0
            data_loader = tqdm(sampler_valid)
            for step, data in enumerate(data_loader):
                _, n_id, adjs = data
                exp = exp_gcn_valid[n_id].to(device)
                edge_index, _, _ = adjs.to(device)
                coordinate_input = coordinates_valid[n_id].to(device)
                edge_weights = compute_edge_weights(coordinate_input, edge_index)
                sample_num += exp.shape[0]
                _, pred, _ = model(exp.to(device), edge_index, edge_weights, coordinate_input)
                pred_classes = torch.max(pred, dim=1)[1]
                label_temp = label_valid[n_id].to(device)
                accu_num += torch.eq(pred_classes, label_temp.to(device)).sum().item()
                loss = loss_function(pred, label_temp.to(device)).item()
                accu_loss += loss
                data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                       accu_loss.item() / (step + 1),
                                                                                       accu_num.item() / sample_num)
                del exp, label_temp, pred, loss, data
                torch.cuda.empty_cache()
            val_loss, val_acc = accu_loss.item() / (step + 1), accu_num.item() / sample_num
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if ((epoch + 1) % 10 == 0):
            if platform.system().lower() == 'windows':
                torch.save(model.state_dict(), project_path+"/model-{}.pth".format(epoch))
            else:
                torch.save(model.state_dict(), "/%s"%project_path+"/model-{}.pth".format(epoch))
    print('Training finished!')


if __name__ == "__main__":
    getVector()


from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Union
import anndata as ad


@contextmanager
def _working_directory(target_directory: Union[str, Path]):
    """Temporarily use the requested output directory for legacy relative paths."""
    resolved_directory = Path(target_directory).expanduser().resolve()
    resolved_directory.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    os.chdir(resolved_directory)
    try:
        yield resolved_directory
    finally:
        os.chdir(previous_directory)


def _read_adata(adata_input, copy: bool = False) -> ad.AnnData:
    """Read an AnnData object or local h5ad path for model training."""
    if isinstance(adata_input, ad.AnnData):
        return adata_input.copy() if copy else adata_input
    input_path = Path(adata_input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return ad.read_h5ad(input_path)


def train_model(
    adata_input,
    project: str,
    gmt_path: str,
    output_dir: Union[str, Path] = ".",
    label_name: str = "Celltype",
    train_batches: Optional[Iterable[Union[str, int]]] = None,
    use_gcn: bool = True,
    **train_kwargs,
) -> Path:
    """Train K-SPADA and return the generated model directory."""
    training_adata = _read_adata(adata_input, copy=False)
    if label_name not in training_adata.obs:
        raise ValueError(f"Missing label column in adata.obs: {label_name}")
    train_batch_list = [] if train_batches is None else [str(value) for value in train_batches]
    if use_gcn and not train_batch_list:
        raise ValueError("train_batches is required when use_gcn=True")
    output_directory = Path(output_dir).expanduser().resolve()
    with _working_directory(output_directory):
        fit_model(
            training_adata,
            gmt_path=gmt_path,
            project=project,
            label_name=label_name,
            use_gcn=use_gcn,
            data_batch=train_batch_list,
            **train_kwargs,
        )
    return output_directory / project
