import torch
import sys
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
from sklearn.manifold import TSNE 
import pandas as pd 
import matplotlib.pyplot as plt 
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
sys.path.append('..')
from utils import *


'''
Macro-consistency for imperceptability
'''
# ------------------------------------ Attribute Visualization ------------------------------------------

def tsne_vis(emb, n_injnode, fig_save_file):
    tsne = TSNE(n_components=2) 
    X_tsne = tsne.fit_transform(emb)

    fig = plt.figure(figsize=(8,8))
    left, bottom, width, height = 0.03,0.03,0.94,0.94
    ax = fig.add_axes([left,bottom,width,height])
    plt.scatter(X_tsne[:-n_injnode,0],X_tsne[:-n_injnode,1], alpha=0.5, edgecolors='white', linewidths=0.8, s=20, label='Original nodes') 
    plt.legend(fontsize=25)
    # fig.savefig(fname=fig_save_file+"origin.pdf",format="pdf")
    plt.scatter(X_tsne[-n_injnode:,0],X_tsne[-n_injnode:,1], alpha=0.5,edgecolors='white',marker='x', c='red',linewidths=3,s=30,label='Injected nodes') 
    plt.legend(fontsize=25,loc='upper left')
    # plt.xticks([])
    # plt.yticks([])
    fig.savefig(fname=fig_save_file+".pdf",format="pdf")
    plt.show()
    return 

def pca_vis(ori_emb, emb, n_injnode, fig_save_file):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(ori_emb)
    x_pca = pca.transform(emb)

    #可视化降维的效果
    fig = plt.figure(figsize=(8,8))
    # left, bottom, width, height = 0.03,0.03,0.94,0.94
    # ax = fig.add_axes([left,bottom,width,height])
    plt.scatter(x_pca[-n_injnode:,0],x_pca[-n_injnode:,1], alpha=0.7,edgecolors='white',marker='x', c='red',linewidths=3,s=30,label='Injected nodes') 
    plt.legend(fontsize=25)
    fig.savefig(fname=fig_save_file+"inj.pdf",format="pdf")
    plt.scatter(x_pca[:-n_injnode,0],x_pca[:-n_injnode,1], alpha=0.3, edgecolors='white', linewidths=0.8, s=20, label='Original nodes') 
    plt.legend(fontsize=25)
    fig.savefig(fname=fig_save_file+".pdf",format="pdf")

    plt.show()
    return

# --------------------------------- Structure Visualization ---------------------------------
def hist_vis(struc, new_struc, fig_save_file):

    fig = plt.figure(figsize=(8,8))
    left, bottom, width, height = 0.07,0.08,0.92,0.9
    ax = fig.add_axes([left,bottom,width,height])
    # sns.set(style="white")
    # plt.boxplot([all_cnt, control_cnt],notch=True)
    ori_wei = np.ones_like(struc)/float(len(struc))
    model_wei = np.ones_like(new_struc)/float(len(new_struc))
    plt.hist((struc,new_struc),bins=50, color=('#1f77b4','red'), weights=(ori_wei,model_wei))
    # plt.hist(new_struc,bins=20, weights=model_wei, alpha=0.5, color='red',label='injected node')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Node Closeness", fontsize=20)
    # plt.ylabel('Probability', fontsize=20)
    plt.legend(labels=['Original nodes','Injected nodes'], fontsize=20)
    plt.grid()
    fig.savefig(fname=fig_save_file+".pdf",format="pdf")
    return



# ---------------------------------- Structure  ------------------------------------
# Structure Rank
def calculate_rank(adj):
    rank = np.linalg.matrix_rank(adj)
    print('Rank of the graph is ', rank)
    return rank

# ---------------------------------- Laplacian Smoothness ----------------------------------
# measurement according to emb
def rayleigh_quotient(adj, emb_matrix, normalization):
    degree = np.diag(adj.sum(1))
    L = degree - adj
    xlx = np.diag((emb_matrix.T.dot(L)).dot(emb_matrix))
    # normalization = 1 / np.sum(emb_matrix**2, axis=0)
    rq = (xlx*normalization).mean()  
    print('Rayleigh quotient of the graph (representation-based):', rq)
    return rq


# ---------------------------------- GraphFD ----------------------------------
# Refer to https://github.com/mseitzer/pytorch-fid
# TODO
def calculate_stat(model, adj, feat, samples):
    emb_all, _ = model(feat, adj)
    emb = emb_all[samples].detach().cpu().numpy()
    mu = np.mean(emb, axis=0)
    sigma = np.cov(emb, rowvar=False)
    # print('mu, sigma', mu, sigma)
    return mu, sigma

def calculate_graphfd(model, ori_adj, ori_feat, new_adj, new_feat, real_sample, fake_sample, eps=1e-6,verbose=True):
    m_real, s_real = calculate_stat(model, ori_adj, ori_feat, real_sample)
    m_fake, s_fake = calculate_stat(model, new_adj, new_feat, fake_sample)
    diff = m_fake - m_real
    covmean, _ = linalg.sqrtm(s_fake.dot(s_real), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(s_fake.shape[0]) * eps
        covmean = linalg.sqrtm((s_fake + offset).dot(s_real + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    graphfd = diff.dot(diff) + np.trace(s_fake) + np.trace(s_real) - 2 * tr_covmean
    if verbose:
        print('GraphFD of the graph is', graphfd)
    return graphfd

def torch_calculate_stat(model, adj, feat, samples):
    emb_all, _ = model(feat, adj)
    emb = emb_all[samples].detach()
    mu = torch.mean(emb, axis=0)
    sigma = torch_cov(emb, rowvar=False)
    # print('mu, sigma', mu, sigma)
    return mu, sigma

def torch_calculate_graphfd(model, ori_adj, ori_feat, new_adj, new_feat, real_sample, fake_sample, eps=1e-6,verbose=True):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/ajbrock/BigGAN-PyTorch
    """
    m_real, s_real = torch_calculate_stat(model, ori_adj, ori_feat, real_sample)
    m_fake, s_fake = torch_calculate_stat(model, new_adj, new_feat, fake_sample)
    assert m_real.shape == m_fake.shape, \
    'Training and test mean vectors have different lengths'
    assert s_real.shape == s_fake.shape, \
    'Training and test covariances have different dimensions'

    diff = m_fake - m_real
    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(s_fake.mm(s_real).unsqueeze(0), 30).squeeze()  
    out = (diff.dot(diff) +  torch.trace(s_fake) + torch.trace(s_real)
            - 2 * torch.trace(covmean))
    return out


def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()+1e-6
        A_fu = torch.where(A.mul(A).sum(dim=1).sum(dim=1)<0)
        normA_nan = torch.isnan(normA)
        normA_0 = torch.isnan(normA)
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
        Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
        for i in range(numIters):
            T = 0.5*(3.0*I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
            if Z.max() > 1e6 or Y.max()>1e6 or T.max()>1e6:
                break
        sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def calculate_attrfd(ori_feature, new_feature, real_sample, fake_sample, eps=1e-6,verbose=True):
    m_real = np.mean(ori_feature[real_sample], axis=0)
    s_real = np.cov(ori_feature[real_sample], rowvar=False)
    m_fake = np.mean(new_feature[fake_sample], axis=0)
    s_fake = np.cov(new_feature[fake_sample], rowvar=False)

    diff = m_fake - m_real
    covmean, _ = linalg.sqrtm(s_fake.dot(s_real), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(s_fake.shape[0]) * eps
        covmean = linalg.sqrtm((s_fake + offset).dot(s_real + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    graphfd = diff.dot(diff) + np.trace(s_fake) + np.trace(s_real) - 2 * tr_covmean
    if verbose:
        print('GraphFD of the graph is', graphfd)
    return graphfd

'''
Micro-consistency for imperceptability
'''

# ---------------------------------- Attribute distance ----------------------------------

def closest_attr(attr_matrix, n, n_inj,normalization):
    attrdis = euclidean_distances(attr_matrix)
    # attrdis = cosine_distances(attr_matrix)
    attrdis[np.eye(attrdis.shape[0],dtype=np.bool)] = 99999
    # if n_injnodes == 1:
    min_dis = attrdis[-n_inj:,:n].min(1).mean()
    min_dis = min_dis * normalization
    # else:
    #     min_dis = attrdis[-n_injnodes:].max(1).mean()
    print('Min Attribute distance between the injected nodes and original nodes:', min_dis)
    return min_dis

def closest_attr_arxiv(attr_matrix, n, inj_idx,normalization):
    attrdis = euclidean_distances(attr_matrix[inj_idx],attr_matrix[:n])
    if inj_idx[0] < n:
        attrdis[:,inj_idx] = 99999
    min_dis = attrdis.min(1).mean()
    min_dis = min_dis * normalization
    print('Min Attribute distance between the injected nodes and original nodes:', min_dis)
    return min_dis


# ---------------------------------- Structure Common Neighbor ---------------------------------
# Common Neighbor
def common_neighb(adj, nodes):
    common_neighb_node_array = []
    adj[np.eye(adj.shape[0],dtype=np.bool)] = 0
    for node in nodes:
        neighbors = adj[node].nonzero()[0]
        edge_bridges = [get_edge_bridgeness(adj, neighb, node) for neighb in neighbors]
        common_neighb_node = np.array(edge_bridges).mean()
        common_neighb_node_array.append(common_neighb_node)
    common_neighb_ratio = np.array(common_neighb_node_array).mean()
    print('Common Neighbor ratio of injected nodes:', common_neighb_ratio)
    return common_neighb_ratio

def get_edge_bridgeness(adj, t1, t2):
    neighb1 = adj[t1].nonzero()[0]
    neighb2 = adj[t2].nonzero()[0]
    edge_bridge = jaccard(neighb1, neighb2)
    return edge_bridge

def jaccard(x,y):
    return len(np.intersect1d(x,y))/len(np.union1d(x,y))

# ----------------------------------- Attributes + Structure -----------------------------------
# Neighbor Attribute Distance
def calculate_neighbdis(attr_matrix, adj, n, n_inj, normalization):
    all_attrdis = euclidean_distances(attr_matrix)
    # all_attrdis = cosine_distances(attr_matrix) 
    if sp.issparse(adj):
        neighb_attrdis = adj.multiply(all_attrdis).tocsr()
    else:    
        neighb_attrdis = all_attrdis * adj
    # neighb_attrdis[np.eye(neighb_attrdis.shape[0],dtype=np.bool)] = 0
    # adj[np.eye(adj.shape[0],dtype=np.bool)] = 0
    neighb_embdis_mean = neighb_attrdis[-n_inj:,:n].sum() / adj[-n_inj:,:n].sum()
    neighb_embdis_mean = neighb_embdis_mean * normalization
    print('Average Inject node neighbor distance of attributes:', neighb_embdis_mean)
    return neighb_embdis_mean

def calculate_neighbdis_arxiv(attr_matrix, adj, n, inj_idx, normalization):
    all_attrdis = euclidean_distances(attr_matrix[inj_idx],attr_matrix[:n])
    # attrdis[:,inj_idx] = 99999
    adj = adj[inj_idx,:n]
    if sp.issparse(adj):
        neighb_attrdis = adj.multiply(all_attrdis).tocsr()
    else:    
        neighb_attrdis = all_attrdis * adj
    neighb_embdis_mean = neighb_attrdis.sum() / adj.sum()
    neighb_embdis_mean = neighb_embdis_mean * normalization
    print('Average Inject node neighbor distance of attributes:', neighb_embdis_mean)
    return neighb_embdis_mean

# ---------------------------------- Learnable Representation Distance ----------------------------------
# LRD
def closest_learnable_rep(emb_matrix, n, n_inj,normalization,verbose=True):
    repdis = euclidean_distances(emb_matrix)
    # repdis = cosine_distances(emb_matrix)
    repdis[np.eye(repdis.shape[0],dtype=np.bool)] = 99999
    # if n_injnodes == 1:
    min_dis = repdis[-n_inj:,:n].min(1).mean()
    min_dis = min_dis * normalization
    # else:
    #     min_dis = repdis[-n_injnodes:].max(1).mean()
    if verbose:
        print('Min Representation distance between the injected nodes and original nodes:', min_dis)
    return min_dis

'''
Diversity for imperceptability
'''
# ------------------------------------ Diversity: Perceptual Similarity ------------------------------------------

def calculate_graphps(inj_idx, all_emb_list, verbose=True):
    n_injnode = len(inj_idx)
    graphps_mx = torch.zeros(n_injnode, n_injnode)
    for layer in range(len(all_emb_list)):
        # emb = all_emb_list[layer].detach().cpu()[-n_injnode:]
        emb = all_emb_list[layer].detach().cpu()[inj_idx]
        norm_emb = F.normalize(emb)
#         bn_emb = bn(emb.cpu()[-n:])
        l2 = euclidean_distances(norm_emb.numpy())
        graphps_mx += l2
    graphps_triu = torch.triu(graphps_mx)
    indices = graphps_triu.nonzero().t()
    graphps = graphps_triu[indices[0],indices[1]].sum() / indices.shape[1]
    if verbose:
        print('Diversity: GraphPS of the graph is', graphps)
    return graphps


def evaluation(new_feat, new_adj_tensor, new_adj, feat, adj_tensor, adj, rep_net, n, fig_save_file, suffix, oriflag=True):
    print('Original Graph:')
    features_np = feat.detach().cpu().numpy()
    all_rep_list = rep_net(feat,adj_tensor)
    rep_np = all_rep_list[1].detach().cpu().numpy()
    norm_xlx = 1 / np.sum(rep_np**2, axis=0)
    norm_attr = 1 / np.linalg.norm(features_np,ord=2, axis=1).mean()
    norm_rep =  1 / np.linalg.norm(rep_np, ord=2, axis=1).mean()
    
    if oriflag:
        # Micro-metric
        attr_sim = closest_attr(features_np, n, n, norm_attr)
        # common_neighbor = common_neighb(adj.toarray(), np.arange(n))
        try:
            rq = rayleigh_quotient(adj, rep_np, norm_xlx)
            neighbor_dis = calculate_neighbdis(features_np, adj.toarray(), n, n, norm_attr)
        except:
            print('pass neighbor_dis')
        lrd = closest_learnable_rep(rep_np, n, n, norm_rep)
        
    else:
        print('New Graph:')
        new_n = new_feat.shape[0]
        new_all_rep_list = rep_net(new_feat, new_adj_tensor)
        new_rep_np = new_all_rep_list[1].detach().cpu().numpy()
        new_feature = new_feat.detach().cpu().numpy()
        
        # tsne_vis(new_feature, new_n-n, fig_save_file+suffix+'_feat')
        # tsne_vis(new_rep_np, new_n-n, fig_save_file+suffix+'_rep')

        
        # Micro-metric
        attr_dis = closest_attr(new_feature, n, new_n-n, norm_attr)
        # common_neighbor = common_neighb(new_adj, np.arange(n, new_n))
        try:
            # rq = rayleigh_quotient(new_adj, new_rep_np, norm_xlx)
            neighbor_dis = calculate_neighbdis(new_feature, new_adj.toarray(), n, new_n-n, norm_attr)
        except:
            print('pass neighbor_dis')
        graph_fd = calculate_graphfd(rep_net, adj_tensor, feat, new_adj_tensor, new_feat, np.arange(n), np.arange(n, new_n))
        # lrd = closest_learnable_rep(new_rep_np, n, new_n-n, norm_rep)

    
        # new_g = nx.from_scipy_sparse_matrix(new_adj.tocsr())
        # new_close = nx.closeness_centrality(new_g)
        # new_array_close = np.array(list(new_close.values()))
        # ori_g = nx.from_scipy_sparse_matrix(adj)
        # close_ori = nx.closeness_centrality(ori_g)
        # array_close_ori = np.array(list(close_ori.values()))
        # hist_vis(array_close_ori, new_array_close, fig_save_file+suffix+'_struct')
        # print('array_close_ori',array_close_ori.mean())
        # print('new_array_close',new_array_close.mean())
    return
