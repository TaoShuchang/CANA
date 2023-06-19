import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import sys
import os,sys

os.chdir(sys.path[0])
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from attacks import *
from gcn_pyg import EGCNGuard,GAT
sys.path.append('../..')
# from gnn_model.EGCNGuard import EGCNGuard as 
from utils import *


def train(model, adj_tensor, feat, labels, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
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

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    row, col = adj_tensor.coalesce().indices()
    adj_tensor = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)).to(device), sparse_sizes=torch.Size(adj.shape),is_sorted=True).to(device)
    # adj_tensor = normalize_tensor(adj_tensor)
    stopper = EarlyStop(patience=500)
    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    # net = EGCNGuard(features.shape[1], 64, labels.max().item() + 1, 2, 0.5,layer_norm_first=args.layer_norm_first, use_ln=args.use_ln).float().to(device)
    net = GAT(features.shape[1], 64, labels.max().item() + 1, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, heads=args.heads).to(device)
    split = np.aload('../../datasets/splits/' + args.dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']



    vals, tests = [], []

    best_val, final_test = 0, 0

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        net.train()
        # print('net.lin_edge',net.lin_edge)
        net(feat,adj_tensor)
        if args.attack == 'vanilla' :
            loss = train(net, adj_tensor, feat, labels, train_mask, optimizer)
        else :
            loss = train_flag(net, adj_tensor, feat, labels, train_mask, optimizer, device, args)

        # if epoch > args.epochs / 2 and epoch % args.test_freq == 0 or epoch == args.epochs:
            # result = test(net, adj_tensor, feat, labels, split_idx, evaluator)
            # _, val, tst = result
        net.eval()
        logits = net(feat, adj_tensor)
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])
        
        if val_acc > best_val :
            best_val = val_acc
            final_test = test_acc
    # acc = accuracy(logits, labels)
        print("Epoch {:05d} | Loss {:.4f} | Train Acc {:.5f} | Val Acc: {:.5f} ".format(
                epoch, loss, train_acc, val_acc))
            # print("Train misclassification {:.4%}".format(1-train_acc))
            # print("Validate misclassification {:.4%}".format(1-val_acc))
            # print("Test misclassification {:.4%}".format(1-test_acc))
        if stopper.step(val_acc, net, net_save_file):   
            break
        # vals.append(best_val)
        # tests.append(final_test)
    
    # print('')
    # print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    # print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FLAG')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
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
    parser.add_argument('--use_ln', type=int,default=1)
    args = parser.parse_args()
    main(args)



'''
CUDA_VISIBLE_DEVICES=3 nohup python -u run_gat_flag.py --attack flag --use_ln 0 --dataset reddit --epochs 1000 --suffix gat > log/reddit/gat_flag.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -u run_gat_flag.py --use_ln 1 --dataset ogbarxiv --epochs 1000 --suffix flag_ln_guard > log/ogbarxiv/flag_ln_guard.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u run_gat_flag.py --use_ln 1 --dataset ogbproducts --epochs 1000 --suffix flag_ln_guard > log/ogbproducts/flag_ln_guard.log 2>&1 &
'''