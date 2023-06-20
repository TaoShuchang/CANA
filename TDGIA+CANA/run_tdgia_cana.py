import torch
import time
import sys
import os
import yaml
import argparse
import numpy as np
import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter

from tdgia import *

sys.path.append('..')
from utils import *
from modules.eval_metric import *
from modules.discriminator import Discriminator
from gnn_model.gin import GIN
from gnn_model.gcn import GCN



def main(args):
    dataset = args.dataset
    suffix = args.suffix
    postprocess = args.postprocess
    postproc_suffix = 'no' if postprocess is False else ''
    print('postprocess',postprocess)
    print('postproc_suffix',postproc_suffix)
    gcn_save_file = '../checkpoint/surrogate_model_gcn/' + dataset + '_partition'
    rep_save_file = '../checkpoint/surrogate_model_gin/' + dataset +'_hmlp'
    graph_save_file = '../attacked_graphs/' + dataset + '/' 
    fig_save_file = 'figures/'+ dataset + '/' + suffix
    yaml_path = '../config.yml'
    writer = SummaryWriter('tensorboard/' + dataset +  '/' + suffix)
    if not os.path.exists(fig_save_file):
        os.makedirs(fig_save_file)
    if not os.path.exists(graph_save_file):
        os.makedirs(graph_save_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = f.read()
        conf_dic = yaml.load(config) 
    adj, features, labels_np = load_npz(f'../datasets/{args.dataset}.npz')

    n = adj.shape[0]
    num_classes = labels_np.max()+1
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    features = features[lcc]
    labels_np = labels_np[lcc]
    n = adj.shape[0]
    print('Nodes num:',n)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)
    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.from_numpy(labels_np).to(device)
    
    split = np.aload('../datasets/splits/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']
    testlabels = labels_np[test_mask]
    # budget = [int(min(degree[i], round(degree.mean()))) for i in test_mask]
    # print(sum(budget))
    feat_lim_max = conf_dic[dataset]['feat_max']
    feat_lim_min = conf_dic[dataset]['feat_min']
    max_connections = conf_dic[dataset]['edge_budget']
    add_num = conf_dic[dataset]['node_budget']
    opt="clip"


    if args.test_rate>0:
        args.test = int(args.test_rate*len(features)/100)
        
    add=int(args.add_rate*0.01*n)
    if add_num>0:
        add=add_num

    surro_net = GCN(features.shape[1], 64, labels_np.max().item() + 1, 0.5).float().to(device)
    surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    rep_net =  GIN(2, 2, features.shape[1], 64, labels_np.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    rep_net.eval()
    surro_net.eval()
    for p in surro_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False

    logits = surro_net(feat, nor_adj_tensor, train_flag=False)

    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    sec = torch.LongTensor(worst_case_class(logits, labels_np)).to(device)
    
    netD = Discriminator(features.shape[1], 64, 1, args.loss_type, 0, args.D_sn, d_type=args.D_type).to(device)
    optimizer_D = torch.optim.Adam([{'params': netD.parameters()}], lr=args.lr_D, weight_decay=args.wd_D)

    new_adj = adj.copy()
    new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
    new_feat = feat.clone()
    num = 0
    print('num',num)
    print('add',add)
    print('opt',opt)
    epoch, out_ep, Dopt_out_ep = 0, 0, 0
    start = time.time()
    while num<add:
        with torch.no_grad():
            curlabels = surro_net(new_feat, new_adj_tensor)

        # Update adj
        addmax = int(add-num)
        if (addmax>add*args.step):
            addmax = int(add*args.step)
        thisadd, adj_new = generateaddon(args.weight1, args.weight2, labels, curlabels, new_adj, n, n+num, test_mask, sconnect=args.connections_self, addmax=addmax,num_classes=num_classes, connect=max_connections)

        # Update feat
        num+=thisadd
        new_adj = adj_new
        print('thisadd', thisadd, ' new_adj.shape', new_adj.shape)
        new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
        
        best_addon, matk, val_GFD, out_ep, Dopt_out_ep = trainaddon(thisadd, args.lr, args.epochs, new_adj_tensor, surro_net, new_feat, adj_tensor, n, test_mask, 
                                                                    labels, sec, netD, rep_net, optimizer_D, args.Dopt, atk_alpha=args.atk_alpha, alpha=args.alpha, beta=args.beta, writer=writer, in_epoch=out_ep, Dopt_in_epoch=Dopt_out_ep)
                                                                                 
        
        print('best_addon',best_addon.shape, best_addon.min(), best_addon.max())
        new_feat = torch.cat((feat, best_addon),0)
        
        writer.add_scalar('Outer Metric/Atk success', matk, epoch)
        writer.add_scalar('Outer Metric/GraphFD', val_GFD, epoch)
        epoch += 1
        
    surro_net.eval()
    new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
    new_feature = new_feat.detach().cpu().numpy()
    # new_feat = new_feat.to(device)
    new_n = new_feat.shape[0]
    
    new_output = surro_net(new_feat, new_nor_adj_tensor)
    atk_suc =((labels[test_mask] != new_output[test_mask].argmax(1)).sum()).item()/test_mask.shape[0]
    print('atk_suc',atk_suc)
    new_adj_sp = sp.csr_matrix(new_adj)
    new_feature_sp = sp.csr_matrix(new_feature)
    dur = time.time() - start
    print('during', dur,'s; ', dur/60, 'm; ', dur/3600, 'h')
    np.savez(graph_save_file + suffix + '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
        adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
        attr_shape=new_feature_sp.shape, labels=labels_np)

    print('New Graph:')
    evaluation(new_feat, new_adj_tensor, new_adj, feat, adj_tensor, adj, rep_net, n, fig_save_file, suffix)
    
    print('*'*15)
    return

if __name__ == '__main__':
    setup_seed(123)

    parser = argparse.ArgumentParser(description="TDGIA")
    parser.add_argument('-f')
    parser.add_argument('--dataset',default='ogbproducts')
    parser.add_argument('--epochs',type=int,default=2001)
    parser.add_argument('--models',nargs='+',default=['gcn_lm'])
    parser.add_argument('--modelsextra',nargs='+',default=[])
    parser.add_argument('--modelseval',nargs='+',default=['gcn_lm','graphsage_norm','sgcn','rgcn','tagcn','appnp','gin'])
    #extra models are only used for label approximation.
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--strategy',default='gia')
    parser.add_argument('--test_rate',type=int,default=0)
    parser.add_argument('--test',type=int,default=50000)
    parser.add_argument('--lr',type=float,default=1)
    parser.add_argument('--step',type=float,default=0.2)
    parser.add_argument('--weight1',type=float,default=0.9)
    parser.add_argument('--weight2',type=float,default=0.1)
    parser.add_argument('--add_rate',type=float,default=1)
    parser.add_argument('--scaling',type=float,default=1)
    parser.add_argument('--opt',default="clip")
    parser.add_argument('--add_num',type=float,default=500)
    parser.add_argument('--connections_self',type=int,default=0)
    parser.add_argument('--postprocess',type=bool,default=True)

    parser.add_argument('--apply_norm',type=int,default=1)
    parser.add_argument('--load',default="default")
    parser.add_argument('--sur',default="default")
    #also evaluate on surrogate models
    parser.add_argument('--save',default="default")
    
    parser.add_argument('--suffix',type=str,default='')
    parser.add_argument('--st_epoch', type=int, default=0, help='number of epochs')
    parser.add_argument('--lr_D', default=1e-3, type=float, help='learning rate of Discriminator')
    parser.add_argument('--Dopt', type=int, default=10000, help='Discriminator optimize Dopt times, G optimize 1 time.')
    parser.add_argument('--D_sn', type=str, default='none', choices=['none', 'SN'] , help='whether Discriminator use spectral_norm')
    parser.add_argument('--D_type', type=str, default='gin_nonorm',  help='Discriminator type')
    parser.add_argument('--loss_type', type=str, default='gan', choices=['gan', 'ns', 'hinge', 'wasserstein'], help='Loss type.')
    parser.add_argument('--step_size', type=int, default=100, help='Step size of the optimizer scheduler')
    parser.add_argument('--patience', type=int, default=500, help='Patience of feature optimization')
    
    parser.add_argument('--atk_alpha', type=float, default=1, help='the coefficient of GAN loss in G loss')
    parser.add_argument('--alpha', type=float, default=1, help='the coefficient of GAN loss in G loss')
    parser.add_argument('--beta', type=float, default=0.01, help='the coefficient of diversity loss in G loss')
    parser.add_argument('--clipfeat', type=bool, default=False, help='the coefficient of GAN when choosing edges to connect')      
    parser.add_argument('--wd_D', type=float, default=0.1)  
    
    args=parser.parse_args()
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    att_sucess = main(args) 


