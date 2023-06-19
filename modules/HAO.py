import os
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn.functional as F
import random
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul

from utils import normalize_tensor

def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

def node_sim_analysis(adj, x):
    # adj = gcn_norm(adj,add_self_loops=False)
    x_neg = adj @ x
    node_sims = F.cosine_similarity(x_neg,x).cpu().numpy()
    return node_sims


def node_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    sims = node_sim_analysis(adj,x)
    if style.lower() == 'random':
        hs = np.random.choice(sims,size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        # mean, var = sims.mean(), sims.var()
        # hs = torch.randn((num,)).to(x.device)
        # hs = mean + hs*torch.pow(torch.tensor(var),0.5)
        from scipy.stats import skewnorm
        a, loc, scale = skewnorm.fit(sims)
        hs = skewnorm(a, loc, scale).rvs(num)
        hs = torch.FloatTensor(hs).to(x.device)
    return hs

def compute_homo_loss(new_adj_tensor, new_feat, inj_feat, n_total):
    with torch.no_grad():
        features_propagate = normalize_tensor(new_adj_tensor) @ new_feat
        features_propagate = features_propagate[n_total:]
    # step 2: calculate the node-centric homophily (here we implement it with cosine similarity)
    homophily = F.cosine_similarity(inj_feat, features_propagate)

    return -homophily.mean()