import torch
import yaml
import time
import sys
import os
import math
import argparse
import numpy as np
import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as Data
np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
from gnia import GNIA

from gnia_cana import NodeInjectAtk
sys.path.append('..')
from modules.eval_metric import *
from utils import *
from utils import _fetch_data
from modules.discriminator import Discriminator
from gnn_model.gin import GIN
from gnn_model.gcn import GCN

setup_seed(123)

def main(opts):
    # hyperparameters
    gpu_id = opts['gpu']
    dataset= opts['dataset']
    suffix = opts['suffix']
    postprocess = opts['postprocess']
    attr_tau = float(opts['attrtau']) if opts['attrtau']!=None else opts['attrtau']
    edge_tau = float(opts['edgetau']) if opts['edgetau']!=None else opts['edgetau']
    lr_G = opts['lr_G']
    lr_D = opts['lr_D']
    patience = opts['patience']
    niterations = opts['niterations']
    batch_size = opts['batchsize']
    Dopt = opts['Dopt']
    
    alpha = opts['alpha']
    beta = opts['beta']
    
    gcn_save_file = '../checkpoint/surrogate_model_gcn/' + dataset + '_partition'
    rep_save_file = '../checkpoint/surrogate_model_gin/' + dataset +'_hmlp'
    fig_save_file = 'figures/'+ dataset + '/' + suffix
    graph_save_file = '../attacked_graphs/' + dataset + '/' + suffix
    
    model_save_file = 'checkpoints/' +  dataset + '/' + suffix
    netD_save_file = 'checkpoints/' + dataset + '/netD_' + suffix
    train_writer = SummaryWriter('tensorboard/' + dataset + '/' + suffix + '/train/')
    val_writer = SummaryWriter('tensorboard/'  + dataset + '/' + suffix + '/val/')
    yaml_path = '../config.yml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = f.read()
        conf_dic = yaml.load(config) 
        # conf_dic = yaml.load(config,Loader = yaml.FullLoader)
    if not os.path.exists(fig_save_file):
        os.makedirs(fig_save_file)
    if not os.path.exists(graph_save_file):
        os.makedirs(graph_save_file)
    if not os.path.exists(netD_save_file):
        os.makedirs(netD_save_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocessing data
    adj, features, labels_np = load_npz(f'../datasets/{dataset}.npz')
    n = adj.shape[0]
    nc = labels_np.max()+1
    nfeat = features.shape[1]
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
    feat_max = conf_dic[dataset]['feat_max']
    feat_min = conf_dic[dataset]['feat_min']
    edge_budget = conf_dic[dataset]['edge_budget']
    node_budget = conf_dic[dataset]['node_budget']
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    eps_threshold = [0 + (50 - 0) * math.exp(-1. * steps / 1) for steps in range(500)]
    
    split = np.aload('../datasets/splits/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']
    print('train_mask', train_mask.shape[0], 'val_mask', val_mask.shape[0], 'test_mask', test_mask.shape[0])
    rep_net =  GIN(2, 2, features.shape[1], 64, labels.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    
    # Surrogate model
    surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))

    # Evalution model
    victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    victim_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    surro_net.eval()
    victim_net.eval()
    rep_net.eval()
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in surro_net.parameters():
        p.requires_grad = False
    for p in rep_net.parameters():
        p.requires_grad = False

    node_emb = surro_net(feat, nor_adj_tensor, train_flag=False)
    W1 = surro_net.gc1.weight.data.detach()
    W2 = surro_net.gc2.weight.data.detach()
    W = torch.mm(W1, W2).t()    

    logits = victim_net(feat, nor_adj_tensor, train_flag=False)

    sec = worst_case_class(logits, labels_np)
    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    all_rep_list = rep_net(feat,adj_tensor)
    rep_np = all_rep_list[1].detach().cpu().numpy()
    features_np = features.toarray()
    norm_rep =  1 / np.linalg.norm(rep_np, ord=2, axis=1).mean()
 
    # Initialization
    model = GNIA(nc, nfeat, W1, W2, postprocess, device, 'gan', feat_max=feat_max, feat_min=feat_min, attr_tau=attr_tau, edge_tau=edge_tau).to(device)

    netD = Discriminator(features.shape[1], 64, 1, 'gan', 0.5, 'none', d_type='gin').to(device)
    stopper = EarlyStop_loss(patience=patience)

    optimizer_D = torch.optim.RMSprop([{'params': netD.parameters()}], lr=lr_D)
    optimizer_G = torch.optim.RMSprop([{'params': model.parameters()}], lr=lr_G, weight_decay=0)
    
    x = torch.LongTensor(train_mask)
    y = labels[train_mask].to(torch.device('cpu'))
    torch_dataset = Data.TensorDataset(x,y)

    num_workers = 16
    batch_loader_G = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=num_workers)
    batch_loader_D = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, num_workers=num_workers)
    iter_loader_D = iter(batch_loader_D)
    iter_loader_G = iter(batch_loader_G)

    # Training and Validation
    val_mask = test_mask
    cur_GFD = 100000
    for iteration in range(niterations):
        training = True
        print("Epoch {:05d}".format(iteration))
        eps = eps_threshold[iteration] if iteration < len(eps_threshold) else eps_threshold[-1]
        for _ in range(Dopt):
            iter_loader_D, real_batch = _fetch_data(iter_dataloader=iter_loader_D, dataloader=batch_loader_D)
            batch_x,batch_y = real_batch

            train_loss_D, train_acc_D = NodeInjectAtk('train_D', model, batch_x, degree, edge_budget, adj, adj_tensor, nor_adj_tensor, 
                                                    victim_net, rep_net, labels, feat, node_emb, sec, W, device, 
                                                    True, training, netD, eps=eps)

            optimizer_D.zero_grad()
            train_loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.1)
            optimizer_D.step()
            train_loss_D_detach = train_loss_D.detach().item()
            del train_loss_D, _
            
        # Generator
        iter_loader_G, real_batch = _fetch_data(iter_dataloader=iter_loader_G, dataloader=batch_loader_G)
        batch_x,batch_y = real_batch
        train_atk_success = np.zeros_like(batch_x)
        # train_cons_GFD = np.zeros_like(batch_x)
        # train_div_ps = np.zeros_like(batch_x)
        train_loss_G_attack = torch.zeros(batch_x.shape)
        train_loss_G_consist = torch.zeros(batch_x.shape)
        train_loss_G_divers = torch.zeros(batch_x.shape)
        train_loss_G = torch.zeros(batch_x.shape)
        i = 0
        for train_batch in batch_x:
        # Update Generator
            matk, latk, lconsist, ldiv = NodeInjectAtk('train_G', model, np.array([train_batch]), degree, edge_budget, adj, adj_tensor, nor_adj_tensor, 
                                                        victim_net, rep_net, labels, feat, node_emb, sec, W, device, 
                                                        True, training, netD, eps=eps)
            loss_G = latk + alpha * lconsist + beta * ldiv

            train_loss_G_attack[i] = latk.item()
            train_loss_G_consist[i] = lconsist.item()
            train_loss_G_divers[i] = ldiv.item()
            train_loss_G[i] = loss_G
            train_atk_success[i] = matk
            i += 1
            del latk, lconsist, ldiv, loss_G
        optimizer_G.zero_grad()
        train_loss_G.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.1)
        optimizer_G.step()

        train_writer.add_scalar('Discriminator/acc D', train_acc_D, iteration)
        train_writer.add_scalar('Discriminator/loss D',train_loss_D_detach, iteration)

        train_writer.add_scalar('loss_G/Attack', train_loss_G_attack.mean().item(), iteration)
        train_writer.add_scalar('loss_G/Consistency', train_loss_G_consist.mean().item(), iteration)
        train_writer.add_scalar('loss_G/Diversity',train_loss_G_divers.mean().item(), iteration)
        train_writer.add_scalar('loss_G/Overall',train_loss_G.mean().item(), iteration)

        train_writer.add_scalar('Metric/Attack Success', train_atk_success.mean(), iteration)
        # train_writer.add_scalar('Metric/Consist GFD', train_cons_GFD.mean(), iteration)
        # train_writer.add_scalar('Metric/Divers GraphPS', train_div_ps.mean(), iteration)

        print('Training set D acc:', train_acc_D)
        print('Training set D loss:', train_loss_D_detach)
        print('Training set G attack loss:', train_loss_G_attack.mean().item())
        print('Training set G consistency loss:', train_loss_G_consist.mean().item())
        print('Training set G diversity loss:', train_loss_G_divers.mean().item())
        print('Training set G loss:', train_loss_G.mean().item())
        print('Training set metric: Attack success rate:', train_atk_success.mean())
        # print('Training set metric: Consistency GFD:', train_cons_GFD.mean())
        # print('Training set metric: Diversity GraphPS:', train_div_ps.mean())
        del train_loss_G_attack, train_loss_G_consist, train_loss_G_divers, train_loss_G

        val_iteration = 10 if iteration < 10 else 1
        if iteration % val_iteration ==0:
            training = False
            matk, latk, lconsist, ldiv, val_loss_D, val_acc_D, val_GFD = NodeInjectAtk('val', model, val_mask, degree, edge_budget, adj, adj_tensor, nor_adj_tensor, 
                                                                victim_net, rep_net, labels, feat, node_emb, sec, W, device, 
                                                                True, training, netD,norm=norm_rep)
            val_loss_G = latk + alpha * lconsist + beta * ldiv
            val_writer.add_scalar('Discriminator/acc D', val_acc_D, iteration)
            val_writer.add_scalar('Discriminator/loss D', val_loss_D, iteration)

            val_writer.add_scalar('loss_G/Attack',latk, iteration)
            val_writer.add_scalar('loss_G/Consistency', lconsist, iteration)
            val_writer.add_scalar('loss_G/Diversity',ldiv, iteration)
            val_writer.add_scalar('loss_G/Overall',val_loss_G, iteration)

            val_writer.add_scalar('Metric/Attack Success', matk, iteration)
            val_writer.add_scalar('Metric/Consist GFD', val_GFD, iteration)
            val_writer.add_scalar('Metric/Divers GraphPS', -ldiv, iteration)
            print('Validation set D acc:', val_acc_D)
            print('Validation set D loss:', val_loss_D)
            print('Validation set G attack loss:', latk)
            print('Validation set G consistency loss:', lconsist)
            print('Validation set G diversity loss:', ldiv)
            print('Validation set G loss:', val_loss_G)
            print('Validation set metric: Attack success rate:', matk)
            # print('Validation set metric: Consistency GFD:', val_GFD.mean())
            # print('Training set metric: Diversity GraphPS:', val_PS)

            stopper.save_checkpoint(model,model_save_file)
            stopper.save_checkpoint(netD,netD_save_file)
            # stopper.save_checkpoint(model,model_save_file + '_' + str(iteration))
            # stopper.save_checkpoint(netD,netD_save_file + '_' + str(iteration))
            del val_loss_G
        torch.cuda.empty_cache()
    train_writer.close()
    val_writer.close()
    
    # TEST
    print('New Graph:')
    target_samples = test_mask
    training = False
    dic = torch.load(model_save_file+'_checkpoint.pt')
    model.load_state_dict(dic)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    new_adj_tensor = adj_tensor.clone()
    new_nor_adj_tensor = nor_adj_tensor.clone()
    new_feat = feat.clone()

    new_feat, new_adj_tensor, new_nor_adj_tensor, atk_suc, _ = NodeInjectAtk('test', model, target_samples, degree, edge_budget, adj, adj_tensor, nor_adj_tensor, 
                                                                victim_net, rep_net, labels, feat, node_emb, sec, W, device, 
                                                                True, training)

    new_adj = sparse_tensor_to_torch_sparse_mx(new_adj_tensor)
    new_feature = new_feat.detach().cpu().numpy()
    print('misclassification', atk_suc)
    inj_feat = new_feat[n:]
    inj_feat = torch.clamp(inj_feat, feat_min, feat_max)
    new_feat = torch.cat((feat, inj_feat), dim=0)
    new_feature = new_feat.detach().cpu().numpy()
    new_logits = victim_net(new_feat, new_nor_adj_tensor)
    metric_atk_suc = ((labels[target_samples] != new_logits[target_samples].argmax(1)).sum()/len(target_samples)).item()

    new_feature = new_feat.detach().cpu().numpy()
    print('misclassification', metric_atk_suc)
    new_adj_sp = sp.csr_matrix(new_adj)
    new_feature_sp = sp.csr_matrix(new_feature)
    np.savez(graph_save_file+ '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
         adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
         attr_shape=new_feature_sp.shape, labels=labels_np)
    print('\t injected feat min, max', new_feature[n:].min(), new_feature[n:].max())
    print('\t injected node budget', new_feature.shape[0] - n)
    print('\t injected edge budget', new_adj_sp[n:].sum(1).max())
    
    evaluation(new_feat, new_adj_tensor, new_adj, feat, adj_tensor, adj, rep_net, n, fig_save_file, suffix, oriflag=False)
    
    print('*'*30)
    


if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='', help='suffix of the checkpoint')
    parser.add_argument('--postprocess', type=bool, default=False, help='whether have sigmoid and rescale')
    parser.add_argument('--dataset', default='ogbproducts', help='dataset to use')
    
    # optimization
    parser.add_argument('--lr_G', default=1e-3, type=float, help='learning rate of Generator')
    parser.add_argument('--lr_D', default=1e-3, type=float, help='learning rate of Discriminator')
    parser.add_argument('--wd', default=0., type=float , help='weight decay')
    
    parser.add_argument('--attrtau', default=None, help='tau of gumbel softmax on attr')
    parser.add_argument('--edgetau', default=0.01, help='tau of gumbel softmax on edge')
    parser.add_argument('--patience', default=100, type=int, help='patience of early stopping')

    parser.add_argument('--niterations', type=int, default=3000, help='number of iterations')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--Dopt', type=int, default=4, help='Discriminator optimize Dopt times, G optimize 1 time.')
    
    parser.add_argument('--alpha', type=float, default=0.5, help='the coefficient of adversarial camouflage loss in G loss')
    parser.add_argument('--beta', type=float, default=0.5, help='the coefficient of diversity sensitive loss in G loss')

    args = parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    att_sucess = main(opts) 

