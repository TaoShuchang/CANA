import argparse
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import sys
import csv
import pandas as pd
from attacks import *
from gcn_pyg import GCN
sys.path.append('../..')
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
    if 'prod' in args.suffix:
        loss, _ = flag_products(model, feat, y,  adj_tensor, args, optimizer, device, F.cross_entropy, train_idx=train_idx)
    else:
        forward = lambda perturb : model(feat+perturb, adj_tensor)[train_idx]
        model_forward = (model, forward)

        loss, _ = flag(model_forward, feat.shape, y, args, optimizer, device, F.cross_entropy)
    

    return loss.item()


def main(args):
    graphpath = '../../' + args.suffix + '_graphs/' + args.dataset + '/' 
    net_save_file = 'checkpoint/' + args.dataset + '/'
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if not os.path.exists(net_save_file):
        os.makedirs(net_save_file)
        
    adj, features, labels_np = load_npz(f'../../datasets/{args.dataset}.npz')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    features = features[lcc]
    labels_np = labels_np[lcc]
    n = adj.shape[0]
    print('Nodes num:',n)

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)
    stopper = EarlyStop(patience=500)
    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    net = GCN(features.shape[1], args.hidden_channels, labels.max().item() + 1, args.num_layers, dropout=args.dropout, layer_norm_first=args.layer_norm_first, use_ln=args.use_ln).float().to(device)

    split = np.aload('../../datasets/splits/' + args.dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        net.train()

        train_loss = train_flag(net, nor_adj_tensor, feat, labels, train_mask, optimizer, device, args)

        net.eval()
        val_loss = train_flag(net, nor_adj_tensor, feat, labels, val_mask, optimizer, device, args)
        logits = net(feat, nor_adj_tensor)
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

    # acc = accuracy(logits, labels)
        print("Epoch {:05d} | Train Loss {:.4f} | Val Loss {:.4f} | Train Acc {:.5f} | Val Acc: {:.5f} ".format(
                epoch, train_loss, val_loss, train_acc, val_acc))
        if 'acc' in args.suffix:
            es = 1-val_acc
        else:
            es = val_loss
        if stopper.step(es, net, net_save_file):   
            break

    net.load_state_dict(torch.load(net_save_file+'flag_checkpoint.pt'))
    net.eval()
    logits = net(feat, nor_adj_tensor)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_mask], labels[val_mask])
    test_acc = accuracy(logp[test_mask], labels[test_mask])
    
    print("Validate accuracy {:.4%}".format(val_acc))
    print("Validate misclassification {:.4%}".format(1-val_acc))
    print("Test accuracy {:.4%}".format(test_acc))
    print("Test misclassification {:.4%}".format(1-test_acc))
    
    graph_save_file = get_filelist(graphpath, [], name='')
    atk_suc_arr = np.zeros(len(graph_save_file))
    graph_name_arr = []
    i = 0
    for graph in graph_save_file:
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        graph_name_arr.append(graph_name)
        adj, features, labels_np = load_npz(graph)
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        nor_adj_tensor = normalize_tensor(adj_tensor)
        feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
        logits = net(feat, nor_adj_tensor)
        atk_suc =((labels[test_mask] != logits[test_mask].argmax(1)).sum()).item()/test_mask.shape[0]
        print("Test misclassification {:.4%}".format(atk_suc))
        atk_suc_arr[i] = atk_suc
        print()
        i += 1
    file = 'logs/' + args.dataset + '.csv'
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.suffix])
    
    dataframe = pd.DataFrame({u'graph_name':graph_name_arr,u'atk_suc':atk_suc_arr})
    dataframe.to_csv(file, mode='a')
    print('*'*30)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='FLAG')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start-seed', type=int, default=0)

    parser.add_argument('--perturb_size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--amp', type=int, default=2)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--attack', type=str, default='flag')
    parser.add_argument('--dataset', type=str, default='ogbproducts')
    parser.add_argument('--suffix', type=str, default='suffix')
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=32)
    args = parser.parse_args()
    print(args)
    main(args)

