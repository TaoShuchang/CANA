import torch
import time
import sys
import os
import math
import argparse
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
sys.path.append('..')
from utils import *
from modules.eval_metric import *


def generate(model, target, degree, edge_budget, adj, adj_tensor,nor_adj_tensor, 
             feat, node_emb, W, ori, best_wrong_label,
             multi, training, device, eps=None):
    target_deg = int(sum([degree[i].item() for i in target]))
    # budget = int(min(target_deg, round(degree.mean()))) if multi else 1
    budget = min(target_deg, edge_budget) if multi else 1
    one_order_nei = adj[target].nonzero()[1]
    tar_norm_adj = nor_adj_tensor[target.item()].to_dense()
    norm_a_target = tar_norm_adj[one_order_nei].unsqueeze(1)
    new_feat, inj_feat, disc_score, masked_score_idx = model(target, one_order_nei, budget, feat, norm_a_target, node_emb,
                                        W[ori], W[best_wrong_label], train_flag=training, eps=eps)

    # new_feat = torch.cat((feat, inj_feat.unsqueeze(0)), 0)
    new_adj_tensor = gen_new_adj_tensor(adj_tensor, disc_score, masked_score_idx, device)
    new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
    return new_feat, disc_score, new_adj_tensor, new_nor_adj_tensor

def percept_loss(all_emb_list, all_emb_list2):
    loss = 0.
    for layer in range(len(all_emb_list)):
        emb = all_emb_list[layer][-1:]
        emb2 = all_emb_list2[layer][-1:]
        norm_emb = F.normalize(emb)
        norm_emb2 = F.normalize(emb2)
        layer_loss = F.pairwise_distance(norm_emb, norm_emb2)
        loss += layer_loss
    return -loss

def compute_G_loss(mode, victim_net, batch, all_emb_list, labels, sec, netD, model, new_feat, new_adj_tensor, new_nor_adj_tensor, all_emb_list2=None):
    batch_size = len(batch)
    new_logits = victim_net(new_feat, new_nor_adj_tensor)
    loss_atk = F.relu(new_logits[batch,labels[batch]] - new_logits[batch,sec[batch]])
    loss_atk = loss_atk.mean()
    # Generator in GAN
    pred_fake_G = netD(new_feat, new_adj_tensor)[1]
    loss_G_fake = model.compute_gan_loss(pred_fake_G[-batch_size:])
    # diversity loss
    if mode == 'train_G':
        metric_atk_suc = int(labels[batch] != new_logits[batch].argmax(1).item())
        loss_G_div = percept_loss(all_emb_list, all_emb_list2)
    else:
        metric_atk_suc = ((labels[batch] != new_logits[batch].argmax(1)).sum()).item()/batch_size
        loss_G_div = -calculate_graphps(batch, all_emb_list, verbose=False)
        
    return metric_atk_suc, loss_atk, loss_G_fake, loss_G_div

def compute_D_loss(batch, netD, adj, adj_tensor, feat, new_feat, new_adj_tensor):
    batch_size = len(batch)
    real_sample = np.random.randint(0, adj.shape[0], (1, batch_size))

    pred_real = netD(feat, adj_tensor)[1]
    pred_fake_D = netD(new_feat.detach(),new_adj_tensor.detach())[1]
    # print('123')
    # print(pred_real[real_sample])
    loss_D = netD.compute_gan_loss(pred_real[real_sample], pred_fake_D[-batch_size:])

    real_label = torch.full((batch_size,1), 1.0, device=pred_real.device)
    fake_label = torch.full((batch_size,1), 0.0, device=pred_real.device)
    acc_real = netD.compute_acc(pred_real[real_sample], real_label) 
    acc_fake = netD.compute_acc(pred_fake_D[-batch_size:], fake_label) 
    acc_D =  (acc_real + acc_fake) / 2

    return loss_D, acc_D


def NodeInjectAtk(mode, model, batch, degree, edge_budget, adj, adj_tensor, nor_adj_tensor, 
                    victim_net, rep_net, labels, feat, node_emb, sec, W, device, 
                    multi, training, netD=None, norm=None, eps=None):
    # new_adj = adj.toarray().copy()
    new_adj_tensor = adj_tensor.clone()
    new_nor_adj_tensor = nor_adj_tensor.clone()
    new_feat = feat.clone()
    n = feat.shape[0]
    # print(' model.add_node_agent', model.add_node_agent.feat_min, model.add_node_agent.feat_max)
    # batch_size = len(batch)
    for node in batch:
        new_feat, _, new_adj_tensor, new_nor_adj_tensor = generate(model, np.array([node]), degree, edge_budget, adj, new_adj_tensor,new_nor_adj_tensor, 
                                                                            new_feat, node_emb, W, labels[node].item(), sec[node],
                                                                            multi, training, device, eps=eps)

    if mode == 'train_G':
        netD.eval()
        all_emb_list = rep_net(new_feat, new_adj_tensor)
        target2 = np.random.randint(adj.shape[0])
        new_feat2, _, new_adj_tensor2, _ = generate(model, np.array([target2]), degree, edge_budget,adj, adj_tensor,nor_adj_tensor, 
                                            feat, node_emb, W, labels[target2].item(), sec[target2],
                                            multi, training, device, eps=eps)
        all_emb_list2 = rep_net(new_feat2, new_adj_tensor2)
        all_emb_list2 = [emb.detach() for emb in all_emb_list2]
        metric_atk_suc, loss_atk, loss_G_fake, loss_G_div = compute_G_loss(mode, victim_net, batch, all_emb_list, labels, sec, netD, model, new_feat, new_adj_tensor, new_nor_adj_tensor, all_emb_list2=all_emb_list2)
        
        
        return metric_atk_suc, loss_atk, loss_G_fake, loss_G_div

    elif mode == 'train_D':
        netD.train()
        loss_D, acc_D = compute_D_loss(batch, netD, adj, adj_tensor, feat, new_feat, new_adj_tensor)
        return loss_D, acc_D

    elif mode == 'val':
        netD.eval()
        all_emb_list = rep_net(new_feat, new_adj_tensor)
        metric_atk_suc, loss_atk, loss_G_fake, loss_G_div = compute_G_loss(mode, victim_net, batch, all_emb_list, labels, sec, netD, model, new_feat, new_adj_tensor, new_nor_adj_tensor)
        loss_D, acc_D = compute_D_loss(batch, netD, adj, adj_tensor, feat, new_feat, new_adj_tensor)
        n = adj_tensor.shape[0]
        new_n = new_adj_tensor.shape[0]
        val_cons_gfd = calculate_graphfd(rep_net, adj_tensor, feat, new_adj_tensor, new_feat, np.arange(n), np.arange(n, new_n))
        # val_minattr = closest_attr(new_feat.detach().cpu().numpy(), n, new_n-n, norm_attr)
        # val_lrd = closest_learnable_rep(all_emb_list[1].detach().cpu().numpy(), n, new_n-n, norm)
        return metric_atk_suc, loss_atk.mean().item(), loss_G_fake.item(), loss_G_div.item(), loss_D.item(), acc_D, val_cons_gfd 

    else:
        # inj_feat = new_feat[n:]
        # inj_feat = torch.clamp(inj_feat, model.add_node_agent.feat_min, model.add_node_agent.feat_max )
        # new_feat = torch.cat((feat, inj_feat), dim=0)
        new_logits = victim_net(new_feat, new_nor_adj_tensor)
        metric_atk_suc = ((labels[batch] != new_logits[batch].argmax(1)).sum()/len(batch)).item()
        return new_feat, new_adj_tensor, new_nor_adj_tensor, metric_atk_suc, new_logits


