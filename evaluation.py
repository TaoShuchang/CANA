import torch
import time
import sys
import os
import math
import argparse
import numpy as np
import scipy.sparse as sp
import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

from modules.eval_metric import *
from utils import *
from gnn_model.gcn import GCN
from gnn_model.gin import GIN

import pynvml
pynvml.nvmlInit()
# GPU的id
setup_seed(123)

def main(opts):
    # hyperparameters
    gpu_id = opts['gpu']
    seed = opts['seed']
    surro_type = opts['surro_type']
    victim_type = opts['victim_type']
    dataset= opts['dataset']
    connect = opts['connect']
    multi = opts['multiedge']
    print("Dataset:",dataset)
    print("Multi:",opts['multiedge'])
    suffix = opts['suffix']
    postprocess = opts['postprocess']
    attr_tau = float(opts['attrtau']) if opts['attrtau']!=None else opts['attrtau']
    edge_tau = float(opts['edgetau']) if opts['edgetau']!=None else opts['edgetau']
    patience = opts['patience']
    best_score = opts['best_score']
    counter = opts['counter']
    niterations = opts['niterations']
    st_iteration = opts['st_iteration']

    loss_type = opts['loss_type']
    D_sn = opts['D_sn']
    D_type = opts['D_type']

    # local_rank = opts['local_rank']
    postproc_suffix = 'no' if postprocess is False else ''
    print('postprocess',postprocess)
    print('postproc_suffix',postproc_suffix)
    gat_save_file = 'checkpoint/surrogate_model_gat/' + dataset + '_soft'
    ppnp_save_file = 'checkpoint/surrogate_model_ppnp/' + dataset + '_app'
    gcn_save_file = 'checkpoint/surrogate_model_gcn/' + dataset + '_partition'
    rep_save_file = 'checkpoint/surrogate_model_gin/'  + dataset + '_hmlp' 
    # fig_save_file = 'figures/' +  dataset + '/' + suffix
    # graph_save_file = 'final_graphs/' + dataset + '/'  + suffix
    # fig_save_file = 'ImpGNIA/figures/' +  dataset + '/' + suffix
    # graph_save_file = 'ImpGNIA/new_graphs/' + dataset + '/'  + suffix
    # print('fig_save_file',fig_save_file)
    # print('graph_save_file',graph_save_file)
    # model_save_file = 'checkpoint/' + surro_type + '_' + postproc_suffix +  'postproc/' + dataset + '_' + suffix
    # netD_save_file = 'checkpoint/' + surro_type + '_netD_atkGAN_'  + postproc_suffix + 'postproc/' + dataset + '_' + 'hmlp_lr1e-4_drop0.5'
    # print(model_save_file)
    # test_writer = SummaryWriter('tensorboard/' + surro_type + '_atkGAN_' + postproc_suffix + 'postproc/test/' + dataset +'/' + suffix)


    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    # Preprocessing data
    adj, features, labels_np = load_npz(f'datasets/{dataset}.npz')
    n = adj.shape[0]
    nc = labels_np.max()+1
    nfeat = features.shape[1]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1
    if connect:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels_np = labels_np[lcc]
        n = adj.shape[0]
        print('Nodes num:',n)

    # adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)

    # mask = np.arange(labels.shape[0])
    # train_mask, val_mask, test_mask = train_val_test_split_tabular(mask, train_size=0.64, val_size=0.16, test_size=0.2, random_state=seed)
    split = np.aload('datasets/splits/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']

    print("Surrogate GNN Model:", surro_type)
    print("Evaluation GNN Model:", victim_type)
    # rep_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    rep_net =  GIN(2, 2, features.shape[1], 64, labels.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    rep_net.eval()
    # Surrogate model
    if surro_type == 'gcn':
        surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
        surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))

    elif surro_type == 'gat':
        surro_net = GAT(num_of_layers=2, num_heads_per_layer=[4, 1], num_features_per_layer=[nfeat, 4, nc],
            add_skip_connection=False, bias=True, dropout=0.1, 
            layer_type=LayerType.IMP2, log_attention_weights=False).to(device)
        surro_net.load_state_dict(torch.load(gat_save_file+'_checkpoint.pt'))
    else:
        prop_appnp = PPRPowerIteration(alpha=0.1, niter=10)
        surro_net = PPNP(nfeat, nc, [64], 0.5, prop_appnp).to(device)
        surro_net.load_state_dict(torch.load(ppnp_save_file+'_checkpoint.pt'))
        
    # Evalution model
    if victim_type == 'gcn':
        victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
        victim_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    elif victim_type == 'gat':
        victim_net = GAT(num_of_layers=2, num_heads_per_layer=[4, 1], num_features_per_layer=[nfeat, 4, nc],
            add_skip_connection=False, bias=True, dropout=0.1, 
            layer_type=LayerType.IMP2, log_attention_weights=False).to(device)
        victim_net.load_state_dict(torch.load(gat_save_file+'_checkpoint.pt'))
    else:
        prop_appnp = PPRPowerIteration(alpha=0.1, niter=10)
        victim_net = PPNP(nfeat, nc, [64], 0.5, prop_appnp).to(device)
        victim_net.load_state_dict(torch.load(ppnp_save_file+'_checkpoint.pt'))
    
    surro_net.eval()
    victim_net.eval()
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False
 
    
    if victim_type == 'gcn':
        logits = victim_net(feat, nor_adj_tensor, train_flag=False)
    elif victim_type == 'gat':
        graph_data = (feat, adj_topo_tensor)
        logits = victim_net(graph_data)[0]
    else:
        logits = victim_net(feat, nor_adj_tensor)
    sec = worst_case_class(logits, labels_np)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    norm_attr = 1 / np.linalg.norm(features.toarray(),ord=2, axis=1).mean()
    # norm_rep =  1 / np.linalg.norm(logits.cpu().numpy(), ord=2, axis=1).mean()
    all_rep_list = rep_net(feat,nor_adj_tensor)
    rep_np = all_rep_list[1].detach().cpu().numpy()
    features_np = features.toarray()
    norm_xlx = 1 / np.sum(rep_np**2, axis=0)
    norm_attr = 1 / np.linalg.norm(features_np,ord=2, axis=1).mean()
    norm_rep =  1 / np.linalg.norm(rep_np, ord=2, axis=1).mean()


    print('New Graph:')
    # target_samples = test_mask
    # training = False
    # graphpath = 'final_graphs/' + dataset + '/' 
    graphpath = 'PGD/new_graphs/' + dataset + '/' 
    # graphpath = 'TDGIA/new_graphs/' + dataset + '/' 
    graph_save_file = get_filelist(graphpath, [], name='_d')
    print('graph_save_file',graph_save_file)
    for graph in graph_save_file:
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        new_adj, new_feature, labels_np = load_npz(graph)
        new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
        # new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
        new_n = new_adj.shape[0]
        new_feat = torch.from_numpy(new_feature.toarray().astype('double')).float().to(device)
        new_feature = new_feature.toarray()
        new_logits = victim_net(new_feat, normalize_tensor(new_adj_tensor))
        metric_atk_suc = ((labels[test_mask] != new_logits[test_mask].argmax(1)).sum()/len(test_mask)).item()
        print('Attack success rate',metric_atk_suc)

        new_all_rep_list = rep_net(new_feat, new_adj_tensor)
        new_rep_np = new_all_rep_list[1].detach().cpu().numpy()
        # pca_vis(features.toarray(), new_feature, new_n-n, fig_save_file+'_feat')
        # pca_vis(rep_np, new_rep_np, new_n-n, fig_save_file+'_rep')
        inj_idx = np.arange(n, new_n)
        print('CAD')
        attr_dis = closest_attr_arxiv(new_feature, n, inj_idx, norm_attr)
        print('Smooth')
        neighbor_dis = calculate_neighbdis_arxiv(new_feature, new_adj, n, inj_idx, norm_attr)
        graph_fd = calculate_graphfd(rep_net, adj_tensor, feat, new_adj_tensor, new_feat, np.arange(n), np.arange(n, new_n))
        # Diversity
        # calculate_graphps(np.arange(n), new_all_rep_list)
        # new_g = nx.from_scipy_sparse_matrix(new_adj)
        # new_close = nx.closeness_centrality(new_g)
        # new_array_close = np.array(list(new_close.values()))[n:]
        # ori_g = nx.from_scipy_sparse_matrix(adj)
        # close_ori = nx.closeness_centrality(ori_g)
        # array_close_ori = np.array(list(close_ori.values()))
        # hist_vis(array_close_ori, new_array_close, fig_save_file+'_struct')
        # print('Original nodes closeness',array_close_ori.mean())
        # print('Injcted nodes closeness',new_array_close.mean())
        print('*'*30)



if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--postprocess', type=bool, default=True, help='whether have sigmoid and rescale')

    # dataset
    parser.add_argument('--dataset', default='citeseer',help='dataset to use')
    
    parser.add_argument('--surro_type', default='gcn',help='surrogate gnn model')
    parser.add_argument('--victim_type', default='gcn',help='evaluation gnn model')

    # optimization
    # parser.add_argument('--optimizer_G', choices=['Adam','SGD', 'RMSprop'], default='RMSprop',help='optimizer_G')
    parser.add_argument('--lr_G', default=0.001, type=float, help='learning rate of Generator')
    parser.add_argument('--lr_D', default=0.001, type=float, help='learning rate of Discriminator')
    parser.add_argument('--wd', default=0., type=float , help='weight decay')
    
    parser.add_argument('--attrtau', default=None, help='tau of gumbel softmax on attr')
    parser.add_argument('--edgetau', default=0.01, help='tau of gumbel softmax on edge')
    
    parser.add_argument('--epsst', default=50, type=int, help='epsilon start: coefficient of the gumbel sampling')
    parser.add_argument('--epsdec', default=1, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--patience', default=50, type=int, help='patience of early stopping')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--multiedge', default=True, type=bool, help='budget of edges connected to injected node')
    parser.add_argument('--allnodes', default=False, type=bool, help='budget of edges connected to injected node')
    
    parser.add_argument('--counter', type=int, default=0, help='counter')
    parser.add_argument('--best_score', type=float, default=0., help='best score')
    parser.add_argument('--st_iteration', type=int, default=0, help='start iteration')
    parser.add_argument('--niterations', type=int, default=100000, help='number of iterations')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--Dopt', type=int, default=1, help='Discriminator optimize Dopt times, G optimize 1 time.')
    parser.add_argument('--D_sn', type=str, default='none', choices=['none', 'SN'] , help='whether Discriminator use spectral_norm')
    parser.add_argument('--D_type', type=str, default='gin', choices=['gin', 'gcn'] , help='Discriminator type')
    parser.add_argument('--loss_type', type=str, choices=['gan', 'ns', 'hinge', 'wasserstein'], help='Loss type.')
    parser.add_argument('--alpha', type=float, default=0.5, help='the coefficient of GAN loss in G loss')
    parser.add_argument('--beta', type=float, default=0.5, help='the coefficient of diversity loss in G loss')
    # parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank')
    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    att_sucess = main(opts) 


'''
CUDA_VISIBLE_DEVICES=3 nohup python -u evaluation.py --suffix final --multiedge True --dataset ogbproducts --gpu 4 > final_graphs/eval_ogbproducts_pgdhao.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u evaluation.py --suffix final --multiedge True --dataset reddit --gpu 6 > final_graphs/eval_reddit.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u evaluation.py --suffix final --multiedge True --dataset reddit --gpu 5 > final_graphs/eval_reddit_pgdhao.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u evaluation.py --suffix final --multiedge True --dataset ogbarxiv --gpu 6 > final_graphs/eval_ogbarxiv_pgdhao.log 2>&1 &

CUDA_VISIBLE_DEVICES=2  nohup python -u evaluation.py --suffix postproc_multi --multiedge True --dataset ogbarxiv --gpu 3 > TDGIA/log/vis/neibdis_ogbarxiv_postproc_multi.log 2>&1 &

CUDA_VISIBLE_DEVICES=1  nohup python -u evaluation.py --suffix hao_pgd --multiedge True --dataset reddit --gpu 3 > PGD/log/vis/neibdis_reddit_hao_pgd.log 2>&1 &

'''
