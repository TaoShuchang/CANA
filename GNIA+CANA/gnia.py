'''
Refer to https://github.com/TaoShuchang/G-NIA
'''

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from modules import losses

def gumbel_softmax(logits, tau, random_flag=False, eps=0, dim=-1):
    if random_flag:
        uniform_rand = torch.rand_like(logits)
        epsilon = torch.zeros_like(uniform_rand) + 1e-6
        nz_uniform_rand = torch.where(uniform_rand<=0, epsilon, uniform_rand)
        gumbels = -(-(nz_uniform_rand.log())).log()    # ~Gumbel(0,1)
        gumbels = (logits + eps * gumbels) / tau  # ~Gumbel(logits,tau)
        output = gumbels.softmax(dim)
    else:
        output = logits/(0.01*tau)
        output = output.softmax(dim)
    return output

def gumbel_topk(logits, budget, tau, random_flag, eps, device):
    mask = torch.zeros(logits.shape,device=device)
    # idx = np.arange(logits.shape[0])
    discrete_score = torch.zeros_like(logits,device=device)
    discrete_score.requires_grad_()
    for i in range(budget):
        # print('disidx:',i)
        if i != 0:
            _, tmp_idx = torch.max(tmp, dim=0)
            mask[tmp_idx] = 9999 
        
        tmp = gumbel_softmax(logits - mask, tau, random_flag, eps)
        discrete_score = discrete_score + tmp
    return discrete_score

# --------------------------- MLP ----------------------------
# MLP
class MLP(nn.Module):
    def __init__(self, input_dim,hid1,hid2,hid3):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_dim, hid1)
        self.l2 = nn.Linear(hid1, hid2)
        self.l3 = nn.Linear(hid2, hid3)


        nn.init.kaiming_normal_(self.l1.weight)
        nn.init.kaiming_normal_(self.l2.weight)
        nn.init.kaiming_normal_(self.l3.weight)


        self.fc1 = nn.Sequential(
            self.l1,
            nn.LeakyReLU(),
            self.l2,
            nn.LeakyReLU(),
            self.l3,
        )
        
    def forward(self, x):
        output = self.fc1(x)
        return output



# ------------------------- Generative model ----------------------------

class AttrGeneration(nn.Module):
    def __init__(self, label_dim, tau, feat_dim, weight1, weight2, postprocess, device,feat_max, feat_min):
        super(AttrGeneration, self).__init__()
        self.weight1 = weight1
        self.weight2 = weight2
        self.postprocess = postprocess
        self.tau = tau
        self.device = device
        self.feat_max = feat_max
        self.feat_min = feat_min
        # direct 方式
        self.obtain_feat = MLP(3*label_dim+2*feat_dim, 128, 512, feat_dim)
    
    def pool_func(self, feat, node_emb, sub_graph_nodes, wlabel, wsec):
        graph_emb = torch.mm(F.relu_(torch.mm(feat[self.target], self.weight1)), self.weight2)
        graph_emb = torch.cat((node_emb[sub_graph_nodes].mean(0).unsqueeze(0), node_emb[self.target], graph_emb, wlabel, wsec), 1)
        return graph_emb

    def forward(self, target, feat, sub_graph_nodes, node_emb, wlabel, wsec, feat_num=None,  eps=1, train_flag=False):
        self.target = target
        # 直接得到属性的方式 
        add_feat = self.obtain_feat(self.pool_func(feat, node_emb, sub_graph_nodes, wlabel, wsec)).squeeze(0)
        if self.postprocess == True:
            inj_feat = add_feat.sigmoid()
            add_feat = (self.feat_max - self.feat_min) * inj_feat + self.feat_min
            # inj_feat = gumbel_topk(self.add_feat, feat_num, self.tau, train_flag, eps, self.device)        
        new_feat = torch.cat((feat, add_feat.unsqueeze(0)), 0)
        return new_feat, add_feat


class EdgeGeneration(nn.Module):
    def __init__(self, label_dim,feat_dim, weight1, weight2, device, tau=None):
        super(EdgeGeneration, self).__init__()
        # TODO 用net.weight
        self.weight1 = weight1
        self.weight2 = weight2
        # self.obtain_score = MLP_edge(5*self.feat_dim+1, 1)
        self.obtain_score = MLP(3*label_dim + 2*feat_dim+1, 512, 32, 1)
        self.tau = tau
        self.device = device
        
    def concat(self, new_feat, adj_tensor, sub_graph_nodes, wlabel, wsec):
        sub_xw = torch.mm(torch.mm(new_feat[sub_graph_nodes], self.weight1), self.weight2)
        tar_xw_rep = (torch.mm(torch.mm(new_feat[self.target], self.weight1), self.weight2)).repeat(len(sub_graph_nodes),1)
        add_xw_rep = (torch.mm(torch.mm(new_feat[-1].unsqueeze(0),  self.weight1), self.weight2)).repeat(len(sub_graph_nodes),1)

        if adj_tensor.is_sparse:
            norm_a_target = adj_tensor[self.target.item()].to_dense()
            norm_a_target = norm_a_target[sub_graph_nodes].unsqueeze(1)
        elif adj_tensor.shape[1] == 1:
            norm_a_target = adj_tensor
        else:
            norm_a_target = adj_tensor[sub_graph_nodes, self.target].unsqueeze(0).t()
        # w_rep = wlabel.repeat(len(self.sub_graph_nodes),1)
        # w_sec_rep = wsec.repeat(len(self.sub_graph_nodes),1)

        concat_output = torch.cat((tar_xw_rep, sub_xw, add_xw_rep, norm_a_target, wlabel.repeat(len(sub_graph_nodes),1), wsec.repeat(len(sub_graph_nodes),1)), 1)
        # concat_output = torch.cat((tar_emb_rep, sub_node_emb, add_emb_rep,tar_add_emb_sub), 1)
        return concat_output

    def forward(self, budget, target, sub_graph_nodes, new_feat, adj_tensor, wlabel, wsec, eps=0, train_flag=False):
        self.budget = budget
        self.target = target
        score = self.obtain_score(self.concat(new_feat, adj_tensor, sub_graph_nodes, wlabel, wsec)).transpose(0,1)
        if score.dim() > 1:
            score = score.squeeze(0)
        elif score.dim() == 0:
            score = score.unsqueeze(0)
        score = gumbel_topk(score, budget, self.tau, train_flag, eps, self.device)
        score_idx = torch.tensor(sub_graph_nodes, dtype=torch.long, device=self.device).unsqueeze(0)
        return score, score_idx


class GNIA(nn.Module):
    def __init__(self, label_dim, feat_dim, weight1, weight2, postprocess,  device, loss_type=None, feat_max=None, feat_min=None, feat_num=None, attr_tau=None, edge_tau=None):
        super(GNIA,self).__init__()
        self.add_node_agent = AttrGeneration(label_dim, attr_tau, feat_dim, weight1, weight2, postprocess, device, feat_max, feat_min)
#         self.add_edge_agent = EdgeGeneration(self.labels, self.budget, tau)
        self.add_edge_agent = EdgeGeneration(label_dim, feat_dim, weight1, weight2, device, edge_tau)
        self.postprocess = postprocess
        self.loss_type = loss_type
        self.device = device
    
    def add_node_and_update(self, feat_num, feat, node_emb, wlabel, wsec, eps=0, train_flag=False):

        return self.add_node_agent(self.target, feat, self.sub_graph_nodes, node_emb, wlabel, wsec, feat_num, eps, train_flag)

    def add_edge_and_update(self, new_feat, nor_adj_tensor, wlabel,wsec, eps=0, train_flag=False):
        
        return self.add_edge_agent(self.budget, self.target, self.sub_graph_nodes, new_feat, nor_adj_tensor, wlabel, wsec, eps, train_flag)

    def forward(self, target, sub_graph_nodes, budget, feat, nor_adj_tensor, node_emb,  
                wlabel, wsec, train_flag, feat_num=None, eps=0):
        self.target = target
        self.sub_graph_nodes = sub_graph_nodes
        self.budget = budget
        wlabel = wlabel.unsqueeze(0)
        wsec = wsec.unsqueeze(0)
        new_feat, add_feat = self.add_node_and_update(feat_num,feat, node_emb, wlabel, wsec, eps, train_flag=train_flag)
        score, masked_score_idx = self.add_edge_and_update(new_feat, nor_adj_tensor, wlabel, wsec, eps=eps, train_flag=train_flag)

        # Evaluation
        # if train_flag:
        #     self.disc_score = self.score
        if not train_flag:
            edge_values, edge_indices = score.topk(budget)
            score = torch.zeros_like(score,device=self.device)
            score[edge_indices]= 1.

        return new_feat, add_feat, score, masked_score_idx


    def compute_gan_loss(self, output):
        r"""
        Computes GAN loss for generator.
        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).
        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        # Compute loss and backprop
        if self.loss_type == "gan":
            lossG = losses.minimax_loss_gen(output)

        elif self.loss_type == "ns":
            lossG = losses.ns_loss_gen(output)

        elif self.loss_type == "hinge":
            lossG = losses.hinge_loss_gen(output)

        elif self.loss_type == "wasserstein":
            lossG = losses.wasserstein_loss_gen(output)

        else:
            raise ValueError("Invalid loss_type {} selected.".format(
                self.loss_type))

        return lossG