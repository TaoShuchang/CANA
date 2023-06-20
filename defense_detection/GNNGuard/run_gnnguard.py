import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import sys
import os,sys

os.chdir(sys.path[0])
from torch_sparse.tensor import SparseTensor
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from guard_pyg import EGCNGuard
sys.path.append('../')
from FLAG.attacks import *
sys.path.append('../..')
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
    graphpath = '../../final_graphs/' + args.dataset + '/' 
    net_save_file = 'checkpoint/' + args.dataset + '/' + args.suffix
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

    stopper = EarlyStop(patience=500)
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


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.epochs + 1):
        net.train()
        # if args.attack == 'flag' :
        train_loss = train(net, adj_tensor, feat, labels, train_mask, optimizer)
        # print('adj_tensor',adj_tensor)
        # loss = train_flag(net, adj_tensor, feat, labels, train_mask, optimizer, device, args)

        val_loss = train(net, adj_tensor, feat, labels, val_mask, optimizer)
        net.eval()
        logits = net(feat, adj_tensor)
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        print("Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Train Acc {:.5f} | Val Acc: {:.5f} ".format(
                epoch, train_loss, val_loss, train_acc, val_acc))
        if 'acc' in args.suffix:
            es = 1-val_acc
        else:
            es = val_loss
        if stopper.step(es, net, net_save_file):   
            break

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
    
    graph_save_file = get_filelist(graphpath, [], name='')
    for graph in graph_save_file:
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        adj, features, labels_np = load_npz(graph)
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
        # adj_tensor = normalize_tensor(adj_tensor)
        row, col = adj_tensor.coalesce().indices()
        adj_tensor = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)), sparse_sizes=torch.Size(adj.shape),is_sorted=True).to(device)
        feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
        logits = net(feat, adj_tensor)
        atk_suc =((labels[test_mask] != logits[test_mask].argmax(1)).sum()).item()/test_mask.shape[0]
        print("Test misclassification {:.4%}".format(atk_suc))
        print()

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

CUDA_VISIBLE_DEVICES=5 nohup python -u run_guard_invi.py --dataset ogbproducts --dropout 0.2 --layer_norm_first --GNNGuard True --epochs 10000 --lr 1e-3 --suffix EGuard_first_lr-3_drop0.2 > log/ogbproducts/Eguard_first_lr-3_drop0.2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -u run_guard_invi.py --dropout 0 --dataset reddit --layer_norm_first --GNNGuard True --epochs 10000 --lr 1e-4 --suffix EGuard_first_lr-4_drop0_1 > log/reddit/Eguard_first_lr-4_drop0_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python -u run_guard_invi.py --dataset ogbarxiv --layer_norm_first --wd 0 --use_ln 1 --GNNGuard True --epochs 10000 --lr 1e-2 --suffix EGuard_first_lr-2_ln_6 > log/ogbarxiv/Eguard_first_lr-2_ln_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u run_guard_invi.py --dataset ogbarxiv --layer_norm_first --wd 0 --use_ln 1 --GNNGuard True --epochs 10000 --lr 1e-2 --dropout 0.6 --suffix EGuard_first_lr-2_ln_drop0.6 > log/ogbarxiv/Eguard_first_lr-2_ln_drop0.6.log 2>&1 &


'''