import argparse
import csv
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import sys
import os,sys

os.chdir(sys.path[0])
from torch_sparse.tensor import SparseTensor
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from guard_pyg import GCN,EGCNGuard
sys.path.append('../')
from FLAG.attacks import *
sys.path.append('../..')
# from gnn_model.gcn import GCN as 
from utils import *


def train(model, adj_tensor, feat, labels, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(feat, adj_tensor)[train_idx]
    loss = F.cross_entropy(out, labels[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def train_flag(model, adj_tensor, feat, labels, train_idx, optimizer, device, args):

    y = labels[train_idx]
    forward = lambda perturb : model(feat+perturb, adj_tensor)[train_idx]
    model_forward = (model, forward)

    loss, _ = flag(model_forward, feat.shape, y, args, optimizer, device, F.cross_entropy)

    return loss.item()


def main(args):
    # graphpath = '../../final_graphs/' + args.dataset + '/' 
    graphpath = '../../PGD/new_graphs/' + args.dataset + '/' 
    # prefix = '/home/taoshuchang/2022_invisibility/defense_models/Guard/'
    net_save_file = 'checkpoint/' + args.dataset + '/' + args.suffix
    # net_save_file = prefix+ net_save_file 
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    adj, features, labels_np = load_npz(f'../../datasets/{args.dataset}.npz')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    features = features[lcc]
    labels_np = labels_np[lcc]
    n = adj.shape[0]
    print('Nodes num:',n)

    # adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
    # adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    # if 'EGuard' in args.suffix:
    net = EGCNGuard(features.shape[1], 128, labels.max().item() + 1, num_layers=args.num_layers, dropout=args.dropout, layer_norm_first=args.layer_norm_first, use_ln=args.use_ln).to(device)
    row, col = adj_tensor.coalesce().indices()
    adj_tensor = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)), sparse_sizes=torch.Size(adj.shape),is_sorted=True).to(device)
    # else:
    #     net = GCN(features.shape[1], 64, labels.max().item() + 1, dropout=0.5,device=device,attention=args.GNNGuard).to(device)

    split = np.aload('../../datasets/splits/' + args.dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']


    net.load_state_dict(torch.load(net_save_file+'_checkpoint.pt'))
    net.eval()
    logits = net(feat, adj_tensor)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_mask], labels[val_mask])
    test_acc = accuracy(logp[test_mask], labels[test_mask])
    
    print("Validate accuracy {:.4%}".format(val_acc))
    print("Validate misclassification {:.4%}".format(1-val_acc))
    print("Test accuracy {:.4%}".format(test_acc))
    print("Test misclassification {:.4%}".format(1-test_acc))
    single_file = []
    if args.dataset == 'ogbproducts':
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_40.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_200.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_480.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_100.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_120.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_140.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_160.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbproducts/alpha10_Dopt100_lr0.01_180.npz')
        single_file.append('../../final_graphs/ogbproducts/TDGIA_DINA_alpha10_Dopt100_lr0.01_100.npz')
    elif args.dataset == 'ogbarxiv':
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/alpha10_Dopt20_lr-3_160.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/alpha10_Dopt20_lr-3_300.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/initfeat_alpha50_lr-3_Dopt1_20.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/initfeat_alpha50_lr-3_Dopt1_40.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/initfeat_alpha50_lr-3_Dopt1_60.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/ogbarxiv/initfeat_alpha50_lr-3_Dopt1_100.npz')
    else:
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha5_Dopt20_lr0.1.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha10_Dopt100_lr0.1.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha5_Dopt100_lr0.1.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha10_Dopt100_lr0.01_60.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha10_Dopt100_lr0.01_100.npz')
        single_file.append('../../TDGIA_DINA/new_graphs/prestruc/reddit/alpha10_Dopt100_lr0.01_200.npz')
    # graph_save_file = get_filelist(graphpath, [], name='')
    # graphpath = '../../PGD_DINA/new_graphs/' + args.dataset + '/candidate/' 
    # graphpath = '../../GNIA_DINA/new_graphs/' + args.dataset + '/new/' 
    # graphpath = '../../TDGIA_DINA/new_graphs/prestruc/' + args.dataset + '/' 
    graph_save_file = get_filelist(graphpath, [], name='_d')
    print('graph_save_file',graph_save_file)
    atk_suc_arr = np.zeros(len(graph_save_file))
    graph_name_arr = []
    i = 0
    for graph in graph_save_file:
    # for graph in single_file:
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        graph_name_arr.append(graph_name)
        adj, features, labels_np = load_npz(graph)
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        # adj_tensor = normalize_tensor(adj_tensor)
        row, col = adj_tensor.coalesce().indices()
        adj_tensor = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)), sparse_sizes=torch.Size(adj.shape),is_sorted=True).to(device)
        feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
        logits = net(feat, adj_tensor)
        atk_suc =((labels[test_mask] != logits[test_mask].argmax(1)).sum()).item()/test_mask.shape[0]
        print("Test misclassification {:.4%}".format(atk_suc))
        print('\t injected feat min, max', features[n:].min(), features[n:].max())
        print('\t injected node budget', features.shape[0] - n)
        print('\t injected edge budget', adj[n:].sum(1).max())
        atk_suc_arr[i] = atk_suc
        print()
        i += 1
        
    file = 'log/' + args.dataset + '/' + args.dataset + '_hao.csv'
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['PGD+HAO'])
    
    #     print()
    dataframe = pd.DataFrame({u'detect_name':graph_name_arr,u'atk_suc':atk_suc_arr})
    dataframe.to_csv(file, mode='a')
    print('*'*30)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FLAG')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_sage', default=False, action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start-seed', type=int, default=0)

    parser.add_argument('--step-size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--attack', type=str, default='flag')
    parser.add_argument('--dataset', type=str, default='ogbproducts')
    parser.add_argument('--suffix', type=str, default='suffix')
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--GNNGuard', type=bool,default=True)
    args = parser.parse_args()
    print(args)
    main(args)



'''

CUDA_VISIBLE_DEVICES=3 nohup python -u eval_attack_invi.py --dataset ogbproducts --layer_norm_first --dropout 0.3 --suffix best_EGuard_first_lr-3_drop0.3 >> log/ogbproducts/eval_best_Eguard_first_lr-3_drop0.3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u eval_attack_invi.py --dataset reddit --layer_norm_first --suffix best_EGuard_first_lr-3_drop0_1 > log/reddit/eval_best_EGuard_first_lr-3_drop0_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u eval_attack_invi.py --dataset reddit --layer_norm_first --suffix best_EGuard_first_lr-3_drop0_1 >> log/reddit/eval_best_EGuard_first_lr-3_drop0_1.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -u eval_attack_invi.py --dataset ogbarxiv --layer_norm_first --wd 0 --use_ln 1 --GNNGuard True --epochs 10000 --lr 1e-2 --suffix best_EGuard_first_lr-2_ln_6 >> log/ogbarxiv/eval_best_Eguard_first_lr-2_ln_6.log 2>&1 & 



'''