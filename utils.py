import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn.functional as F
import random
# from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def _fetch_data(iter_dataloader, dataloader):
    """
    Fetches the next set of data and refresh the iterator when it is exhausted.
    Follows python EAFP, so no iterator.hasNext() is used.
    """
    try:
        real_batch = next(iter_dataloader)
    except StopIteration:
        iter_dataloader = iter(dataloader)
        real_batch = next(iter_dataloader)

    return iter_dataloader, real_batch


def save_weights_checkpoint(weights,file):
    '''Saves model when validation loss decrease.'''
    torch.save(weights, file+'_checkpoint.pt')

def save_checkpoint(model,file):
    '''Saves model when validation loss decrease.'''
    torch.save(model.state_dict(), file+'_checkpoint.pt')

class EarlyStop_loss:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score, model, file, atk, atk_stand, netD=None, netD_save_file=None):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model,file)
            if netD is not None:
                self.save_checkpoint(netD,netD_save_file)
        elif np.isnan(score):
            print('GFD is Nan')
            self.early_stop = True
        elif score <= self.best_score and atk >= atk_stand:
                self.best_score = score
                self.save_checkpoint(model,file)
                if netD is not None:
                    self.save_checkpoint(netD,netD_save_file)
                self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file+'_checkpoint.pt')  

class EarlyStop_tdgia_loss:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score, adj,feature,label, graph_file, atk, atk_stand, netD=None, netD_save_file=None):
        if self.best_score is None:
            self.best_score = score
            self.save_graph(adj,feature,label, graph_file)
            if netD is not None:
                save_checkpoint(netD,netD_save_file)
        elif score <= self.best_score and atk >= atk_stand:
            self.best_score = score
            self.save_graph(adj,feature,label,graph_file)
            if netD is not None:
                save_checkpoint(netD,netD_save_file)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop 

    def save_graph(self,adj, feature, labels_np, graph_save_file):
        new_adj_sp = sp.csr_matrix(adj)
        new_feature_sp = sp.csr_matrix(feature.detach().cpu().numpy())
        np.savez(graph_save_file + '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
            adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
            attr_shape=new_feature_sp.shape, labels=labels_np)
    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file+'_checkpoint.pt')  

# defense
class EarlyStop:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score, model, file):
        # print('atk',atk, atk_stand)
        # print('score',score,self.best_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model,file)
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(model,file)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file+'_checkpoint.pt')  

# defense
class EarlyStop_defense:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score, model, file):
        # print('atk',atk, atk_stand)
        # print('score',score,self.best_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model,file)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model,file)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, file):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file+'_checkpoint.pt')  
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True     

def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
        
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def train_val_test_split_tabular(arrays, train_size=0.1, val_size=0.1, test_size=0.8, stratify=None, random_state=123):

    # if len(set(array.shape[0] for array in arrays)) != 1:
    #     raise ValueError("Arrays must have equal first dimension.")
    # idx = np.arange(arrays[0].shape[0])
    idx = arrays
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


# --------------------- Load data ----------------------

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.aload(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


# ------------------------ Normalize -----------------------
# D^(-0.5) * A * D^(-0.5)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def dense_normalize(mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx

def normalize_tensor(sp_adj_tensor,edges=None, sub_graph_nodes=None,sp_degree=None,eps=1e-4):
    edge_index = sp_adj_tensor.coalesce().indices()
    edge_weight = sp_adj_tensor.coalesce().values()
    shape = sp_adj_tensor.shape
    num_nodes= sp_adj_tensor.size(0)

    row, col = edge_index
    if sp_degree is None:
        # print('None')
        deg = torch.sparse.sum(sp_adj_tensor,1).to_dense().flatten()
    else:
        # print('sp')
        deg = sp_degree
        for i in range(len(edges)):
            idx = sub_graph_nodes[0,i]
            deg[idx] = deg[idx] + edges[i]
        last_deg = torch.sparse.sum(sp_adj_tensor[-1]).unsqueeze(0).data
        deg = torch.cat((deg,last_deg))
        
    deg_inv_sqrt = (deg + eps).pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    nor_adj_tensor = torch.sparse.FloatTensor(edge_index, values, shape)
    del edge_index, edge_weight, values, deg_inv_sqrt
    return nor_adj_tensor

def sparse_mx_to_torch_sparse_tensor(sparse_mx,device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape,dtype=torch.float,device=device)

def sparse_tensor_to_torch_sparse_mx(sparse_tensor,device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    row = sparse_tensor.coalesce().indices()[0].detach().cpu().numpy()
    col = sparse_tensor.coalesce().indices()[1].detach().cpu().numpy()
    data = sparse_tensor.coalesce().values().detach().cpu().numpy()
    shape = (sparse_tensor.shape[0],sparse_tensor.shape[1])
    return sp.coo_matrix((data, (row, col)),shape=shape)


# --------------------------------- Sub-graph ------------------------ 

def k_order_nei(adj, k, target):
    for i in range(k):
        if i == 0:
            one_order_nei = adj[target].nonzero()[1]
            sub_graph_nodes = one_order_nei
        else:
            sub_graph_nodes = np.unique(adj[sub_graph_nodes].nonzero()[1])
        
    sub_tar = np.where(sub_graph_nodes==target)[0]
    sub_idx = np.where(np.in1d(sub_graph_nodes, one_order_nei))[0]
    return one_order_nei, sub_graph_nodes, sub_tar, sub_idx


def sub_graph_tensor(two_order_nei, feat, adj, normadj, device):
    sub_feat = feat[two_order_nei]
    sub_adj = adj[two_order_nei][:, two_order_nei]
    sub_nor_adj = normadj[two_order_nei][:, two_order_nei]
    sub_adj_tensor = sparse_mx_to_torch_sparse_tensor(sub_adj,device=device).to(device)
    sub_nor_adj_tensor = sparse_mx_to_torch_sparse_tensor(sub_nor_adj,device=device).to(device)

    return sub_feat, sub_adj_tensor, sub_nor_adj_tensor


# -------------------------------------- After Attack ----------------------------------

def gen_new_adj_tensor(adj_tensor, edges, edge_idx, device):
    # sparse tensor
    n = adj_tensor.shape[0]
    sub_mask_shape = edge_idx.shape
    extend_i0 = torch.cat((torch.full(sub_mask_shape,n,dtype=torch.long,device=device), edge_idx), 0)
    extend_i1 = torch.cat((edge_idx, torch.full(sub_mask_shape,n,dtype=torch.long,device=device)), 0)
    extend_i = torch.cat((extend_i0, extend_i1), 1)
    
    # add_one = torch.ones(1,device=device)
    extend_v = torch.cat((edges, edges, torch.ones(1,device=device)),0)
    extend_v = torch.cat((edges, edges),0)

    i = adj_tensor._indices()
    v = adj_tensor._values()
        
    new_i = torch.cat([i, extend_i], 1)
    new_v = torch.cat([v, extend_v], 0)
    new_adj_tensor = torch.sparse.FloatTensor(new_i, new_v, torch.Size([n+1,n+1]))
    return new_adj_tensor


def tdgia_gen_new_adj_tensor(adj_tensor, testindex, device):
    # sparse tensor
    n = adj_tensor.shape[0]
    
    num_test = len(testindex)
    print('num_test',num_test)
    inj = torch.arange(n,n+num_test,device=device).unsqueeze(0)
    ori = torch.tensor(testindex,device=device).unsqueeze(0)
    extend_row = torch.cat((inj, ori), 0)
    extend_col = torch.cat((ori, inj), 0)
    extend_addone = torch.cat((inj, inj), 0)
    extend_i = torch.cat((extend_row, extend_col, extend_addone), 1)
    # add_one = torch.ones(1,device=device)
    edges = torch.full((num_test,),1.0, device=device)
    extend_v = torch.cat((edges, edges, torch.ones(num_test,device=device)), 0)

    i = adj_tensor._indices()
    v = adj_tensor._values()   
    new_i = torch.cat([i, extend_i], 1)
    new_v = torch.cat([v, extend_v], 0)
    new_adj_tensor = torch.sparse.FloatTensor(new_i, new_v, torch.Size([n+num_test,n+num_test]))
    return new_adj_tensor



def block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_feat, inj_feat, W):
    if inj_feat.dim() == 1:
        inj_feat = inj_feat.unsqueeze(0)
    ori_xw = torch.mm(ori_feat, W)
    inj_xw = torch.mm(inj_feat, W)
    ori_adj_ori_xw = torch.sparse.mm(ori_adj_tensor, ori_xw)
    injrow_adj_ori_xw = torch.sparse.mm(inj_adj_tensor_row, ori_xw)
    injcol_adj_inj_xw = torch.sparse.mm(inj_adj_tensor_col, inj_xw)
    inj_adj_inj_xw = torch.sparse.mm(inj_adj_tensor_one, inj_xw)
    
    ori_emb = ori_adj_ori_xw + injcol_adj_inj_xw
    inj_emb = injrow_adj_ori_xw + inj_adj_inj_xw
    return ori_emb, inj_emb

def approximate_evaluate_res(degree, ori_adj_tensor, ori_feat, edges, edge_idx, inj_feat, W1, W2, budget, device):
    # inject adj tensor
    n = ori_adj_tensor.shape[0]
    zeros = torch.zeros(edge_idx.shape).long()
    extend_i_row = torch.cat((zeros, edge_idx), 0).to(device)
    extend_i_col = torch.cat((edge_idx, zeros), 0).to(device)

    r_inv = np.power(budget + 1, -0.5)
    edge_idx_0 = edge_idx[0]
    sub_d = degree[edge_idx_0] + 1
    sub_d_inv = torch.pow(sub_d, -0.5)
    extend_v = sub_d_inv * edges * r_inv
    add_one = torch.FloatTensor([r_inv*r_inv])

    inj_adj_tensor_row = torch.sparse.FloatTensor(extend_i_row, extend_v, torch.Size([1, n]))
    inj_adj_tensor_col = torch.sparse.FloatTensor(extend_i_col, extend_v, torch.Size([n, 1]))
    inj_adj_tensor_one = torch.sparse.FloatTensor(torch.LongTensor([[0],[0]]), add_one, torch.Size([1, 1])).to(device)
    
    ori_emb1, inj_emb1 = block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_feat, inj_feat, W1)
    ori_emb1 = F.relu(ori_emb1)
    inj_emb1 = F.relu(inj_emb1)
    ori_emb2, inj_emb2 = block_spmm(ori_adj_tensor, inj_adj_tensor_row, inj_adj_tensor_col, inj_adj_tensor_one, ori_emb1, inj_emb1, W2)

    approimate_emb = torch.cat((ori_emb2, inj_emb2))
    return approimate_emb


def gen_new_adj_topo_tensor(adj_topo_tensor, edges, sub_graph_nodes, device):
    # tensor
    n = adj_topo_tensor.shape[0]
    new_edge = torch.zeros((1,n)).to(device)
    new_edge[0, sub_graph_nodes] = edges
    new_adj_topo_tensor = torch.cat((adj_topo_tensor, new_edge),dim=0)
    add_one = torch.ones((1,1)).to(device)
    new_inj_edge = torch.cat((new_edge, add_one), dim=1)
    new_adj_topo_tensor = torch.cat((new_adj_topo_tensor, new_inj_edge.reshape(n+1,1)),dim=1)
    return new_adj_topo_tensor

def gen_new_edge_idx(adj_edge_index, disc_score, masked_score_idx, device):
    inj_node = adj_edge_index.max() + 1
    inj_sub_idx = torch.where(disc_score>=0.9)[0]
    inj_edge_idx = masked_score_idx[0,inj_sub_idx].unsqueeze(0)
    inj_idx = inj_node.repeat(inj_edge_idx.shape)
    pos_inj_edges = torch.cat((inj_idx, inj_idx),dim=0).to(device)
    rev_inj_edges = torch.cat((inj_idx, inj_idx),dim=0).to(device)
    new_edge_idx = torch.cat((adj_edge_index, pos_inj_edges, rev_inj_edges),dim=1)
    return new_edge_idx



def worst_case_class(logp, labels_np):
    logits_np = logp.cpu().numpy()
    max_indx = logits_np.argmax(1)
    for i, indx in enumerate(max_indx):
        logits_np[i][indx] = np.nan
        logits_np[i][labels_np[i]] = np.nan
    second_max_indx = np.nanargmax(logits_np, axis=1)

    return second_max_indx


def update_optimizer(optimizer, gamma, cur_step, step):
    """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
    MAX_LR = 1
    MIN_LR = 1e-5
    cur_lr = optimizer.param_groups[0]['lr']
    estimated_lr = cur_lr
    if cur_step % step == 0:
        estimated_lr = cur_lr * gamma
        
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr
    if cur_step % step == 0:
        print('Update LR',estimated_lr, lr)
    # set LR
    for parame_group in optimizer.param_groups:
        parame_group["lr"] = lr

    return optimizer

def propagation_matrix(adj, alpha=0.85, sigma=1):
    """
    Computes the propagation matrix  (1-alpha)(I - alpha D^{-sigma} A D^{sigma-1})^{-1}.

    Parameters
    ----------
    adj : tensor, shape [n, n]
    alpha : float
        (1-alpha) is the teleport probability.
    sigma
        Hyper-parameter controlling the propagation style.
        Set sigma=1 to obtain the PPR matrix.
    Returns
    -------
    prop_matrix : tensor, shape [n, n]
        Propagation matrix.
    """
    deg = adj.sum(1)
    deg_min_sig = torch.matrix_power(torch.diag(deg), -sigma)
    # 为了节省内存 100m
    if sigma - 1 == 0:
        deg_sig_min = torch.diag(torch.ones_like(deg))
    else:
        deg_sig_min = torch.matrix_power(torch.diag(deg), sigma - 1)

    n = adj.shape[0]
    pre_inv = torch.eye(n,device=adj.device) - alpha * deg_min_sig @ adj @ deg_sig_min

    prop_matrix = (1 - alpha) * torch.inverse(pre_inv)
    del pre_inv,deg_min_sig, adj
    return prop_matrix


def get_filelist(cur_dir, Filelist, name=''):
    newDir = cur_dir
    if os.path.isfile(cur_dir) and name in cur_dir:
        Filelist.append(cur_dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(cur_dir):
        for s in os.listdir(cur_dir):
            # 如果需要忽略某些文件夹，使用以下代码
            if "others" in s:
                continue
            newDir=os.path.join(cur_dir,s)
            get_filelist(newDir, Filelist, name)
    return Filelist