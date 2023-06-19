import torch
import time
import yaml
import sys
import os
import math
import argparse
from torch_sparse import SparseTensor
import networkx as nx
import numpy as np
import scipy.sparse as sp
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans, Birch, AgglomerativeClustering,SpectralClustering
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
import os,sys

os.chdir(sys.path[0])
import os,sys

sys.path.append("./")

from attacks.pgd import PGD
sys.path.append('..')
from modules.eval_metric import *
from utils import *
from gnn_model.gin import GIN
from gnn_model.gcn import GCN
from defense_models.AttackDetect.eval_classifier_old import evaluate_atk
import pynvml
pynvml.nvmlInit()
# GPU的id


def main(opts):
    # hyperparameters
    surro_type = opts['surro_type']
    victim_type = opts['victim_type']
    dataset= opts['dataset']
    connect = opts['connect']
    print("Dataset:",dataset)
    print("Multi:",opts['multiedge'])
    suffix = opts['suffix']
    postprocess = opts['postprocess']

    # local_rank = opts['local_rank']
    yaml_path = '../config.yml'
    gcn_save_file = '../checkpoint/surrogate_model_gcn/' + dataset + '_partition'
    rep_save_file = '../checkpoint/surrogate_model_gin/' + dataset +'_hmlp'
    fig_save_file = 'figures/'+ dataset + '/' 
    graph_save_file = 'new_graphs/' + dataset + '/'
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = f.read()
        conf_dic = yaml.load(config) 
    # model_save_file = 'checkpoint/' +  dataset + '/repadj_' + suffix
    if not os.path.exists(fig_save_file):
        os.makedirs(fig_save_file)
    if not os.path.exists(graph_save_file):
        os.makedirs(graph_save_file)
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl') 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


    # Preprocessing data
    adj, features, labels_np = load_npz(f'../datasets/{dataset}.npz')
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
    feat_lim_max = conf_dic[dataset]['feat_max']
    feat_lim_min = conf_dic[dataset]['feat_min']
    edge_budget = conf_dic[dataset]['edge_budget']
    node_budget = conf_dic[dataset]['node_budget']
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    print('feat_lim_max',feat_lim_max,'feat_lim_min',feat_lim_min)
    
    # mask = np.arange(labels.shape[0])
    # train_mask, val_mask, test_mask = train_val_test_split_tabular(mask, train_size=0.64, val_size=0.16, test_size=0.2, random_state=seed)
    split = np.aload('../datasets/splits/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']
    index_train = torch.LongTensor(train_mask).to(device)
    index_val = torch.LongTensor(val_mask).to(device)
    index_test = torch.LongTensor(test_mask).to(device)
    node_budget = test_mask.shape[0]

    print("Surrogate GNN Model:", surro_type)
    print("Evaluation GNN Model:", victim_type)
    # rep_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    rep_net =  GIN(2, 2, features.shape[1], 64, labels.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    
    # Surrogate model
    surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    victim_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))

    surro_net.eval()
    victim_net.eval()
    rep_net.eval()
    for p in surro_net.parameters():
        p.requires_grad = False
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in rep_net.parameters():
        p.requires_grad = False
 
    
    logits = victim_net(feat, nor_adj_tensor, train_flag=False)
    sec = torch.LongTensor(worst_case_class(logits, labels_np)).to(device)

    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    
    
    attacker = PGD(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= node_budget,
                 n_edge_max= edge_budget,
                 feat_lim_min=feat_lim_min,
                 feat_lim_max=feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe)
   
    
    # Initialization
    # new_row = adj_tensor.coalesce().indices()[0]
    # new_col = adj_tensor.coalesce().indices()[1]
    # value = adj_tensor.coalesce().values()
    # adj_st = SparseTensor(row=new_row, col=new_col, value=value)
    # new_adj_tmp = adj_st
    # new_feat_tmp = feat
    start = time.time()
    new_adj_tensor, inj_feat = attacker.attack(model=surro_net,
                                            adj=adj,
                                            features=feat,
                                            target_idx=index_test,
                                            labels=labels, sec=sec)
    new_feat = torch.cat([feat, inj_feat])
    
    # new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj_tmp).to(device)
    new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
    new_adj = sparse_tensor_to_torch_sparse_mx(new_adj_tensor)

    new_logits = victim_net(new_feat, new_nor_adj_tensor)
    atk_suc = ((labels[test_mask] != new_logits[test_mask].argmax(1)).sum()/len(test_mask)).item()
    new_n = new_adj.shape[0]
    new_feature = new_feat.detach().cpu().numpy()
    print('Attack success rate:', atk_suc)
    new_adj_sp = sp.csr_matrix(new_adj)
    new_feature_sp = sp.csr_matrix(new_feature)
    dur = time.time() - start
    print('during', dur,'s; ', dur/60, 'm; ', dur/3600, 'h')
    np.savez(graph_save_file+suffix + '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
         adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
         attr_shape=new_feature_sp.shape, labels=labels_np)
    
    evaluation(new_feat, new_adj_tensor, new_adj, feat, adj_tensor, adj, rep_net, n, fig_save_file, suffix)
    
    print('*'*30)
    


if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--postprocess', type=bool, default=False, help='whether have sigmoid and rescale')

    # dataset
    parser.add_argument('--dataset', default='ogbproducts',help='dataset to use')
    
    parser.add_argument('--surro_type', default='gcn',help='surrogate gnn model')
    parser.add_argument('--victim_type', default='gcn',help='evaluation gnn model')

    # optimization
    # parser.add_argument('--optimizer_G', choices=['Adam','SGD', 'RMSprop'], default='RMSprop',help='optimizer_G')
    parser.add_argument('--lr_G', default=1e-5, type=float, help='learning rate of Generator')
    parser.add_argument('--lr_D', default=1e-5, type=float, help='learning rate of Discriminator')
    parser.add_argument('--wd', default=0., type=float , help='weight decay')
    
    parser.add_argument('--attrtau', default=None, help='tau of gumbel softmax on attr')

    parser.add_argument('--patience', default=8000, type=int, help='patience of early stopping')
    parser.add_argument('--atk_stand', default=0.6, type=float, help='attack success standard')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--multiedge', default=False, type=bool, help='budget of edges connected to injected node')
    parser.add_argument('--feat_lim_min', type=float, default=-50)
    parser.add_argument('--feat_lim_max', type=float, default=50)
    # attack feature update epochs
    parser.add_argument('--attack_epoch', type=int, default=800)
    # attack step size
    parser.add_argument('--attack_lr', type=float, default=0.01)
    # early stopping feat upd for attack
    parser.add_argument('--early_stop', type=int, default=1500)
    # weight of the disguised regularization term
    # when it's set to 0, it's equivalent to pgd
    parser.add_argument('--disguise_coe', type=float, default=0.0)
    parser.add_argument('--hinge', action="store_true")
    # maximum number of injected nodes at 'full' data mode
    # if in other data modes, e.g., 'easy', it shall be 1/3 of that in 'full' mode
    parser.add_argument('--n_inject_max', type=int, default=60)
    # maximum number of edges of the injected (per) node 
    parser.add_argument('--n_edge_max', type=int, default=20)
    parser.add_argument('--injnode_n', type=int, default=3000)
    
    # parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank')
    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    att_sucess = main(opts) 

'''
CUDA_VISIBLE_DEVICES=2 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd_time --attack_epoch 1500  --disguise_coe 0 > log/ogbproducts/pgd_time.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd+hao_time --attack_epoch 1500  --disguise_coe 1 > log/ogbproducts/pgd+hao_time.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u run_pgd.py --dataset reddit --suffix pgd_time --attack_epoch 1500  --disguise_coe 0 > log/reddit/pgd_time.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -u run_pgd.py --dataset reddit --suffix pgd+hao_time --attack_epoch 1500  --disguise_coe 1 > log/reddit/pgd+hao_time.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd_time --attack_epoch 1500  --disguise_coe 0 > log/ogbarxiv/pgd_time.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd+hao_time --attack_epoch 1500  --disguise_coe 1 > log/ogbarxiv/pgd+hao_time.log 2>&1 &

'''


'''


nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd --gpu 3 --attack_epoch 1500  --disguise_coe 0 > log/ogbproducts/pgd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd+hao --gpu 6 --attack_epoch 1500  --disguise_coe 1 > log/ogbproducts/pgd+hao.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd+hao_d1 --gpu 6 --attack_epoch 1500  --disguise_coe 1 > log/ogbproducts/pgd+hao_d1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u run_pgd.py --dataset reddit --suffix pgd --gpu 6 --attack_epoch 1500  --disguise_coe 0 > log/reddit/pgd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pgd.py --dataset reddit --suffix pgd+hao --gpu 6 --attack_epoch 1500  --disguise_coe 1 > log/reddit/pgd+hao.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u run_pgd.py --dataset reddit --suffix pgd+hao_d1 --gpu 6 --attack_epoch 1500  --disguise_coe 1 > log/reddit/pgd+hao_d1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd --gpu 5 --attack_epoch 500  --disguise_coe 0 > log/ogbarxiv/pgd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd+hao --gpu 3 --attack_epoch 500  --disguise_coe 1 > log/ogbarxiv/pgd+hao.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd+hao_d1 --gpu 6 --attack_epoch 1500  --disguise_coe 1 > log/ogbarxiv/pgd+hao_d1.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -u run_pgd.py --suffix pgd_1_same --attack_epoch 500 --dataset ogbarxiv--disguise_coe 0 > log/ogbarxiv/pgd_1_same.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u run_pgd.py --suffix pgd+hao_1_same --attack_epoch 500  --disguise_coe 1 --dataset ogbarxiv  > log/ogbarxiv/pgd+hao_1_same.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -u run_pgd.py --dataset ogbarxiv --suffix pgd+hao_d10 --gpu 6 --attack_epoch 1500  --disguise_coe 10 > log/ogbarxiv/pgd+hao_d10.log 2>&1 &


30inj
CUDA_VISIBLE_DEVICES=2 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd_30inj --gpu 2 --attack_epoch 1500  --disguise_coe 0 > log/ogbproducts/pgd_30inj.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd+hao_30inj  --attack_epoch 1500  --disguise_coe 1 > log/ogbproducts/pgd+hao_30inj.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u run_pgd.py --dataset ogbproducts --suffix pgd+hao_30inj_minus  --attack_epoch 1500  --disguise_coe 1 > log/ogbproducts/pgd+hao_30inj_minus.log 2>&1 &
'''
