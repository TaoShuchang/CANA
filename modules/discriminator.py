import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import dgl
from modules import losses
from utils import *
from gnn_model.gcn import GraphConvolution, GCN
from gnn_model.gin_nonorm import GIN as GIN_nonorm
from gnn_model.gin import GIN
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
setup_seed(123)

def _add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, _add_sn(layer))
        
        if isinstance(m, (nn.Linear, GraphConvolution)):
            return nn.utils.spectral_norm(m)
        else:
            return m
   

class Discriminator(nn.Module):
    def __init__(self, nfeat, nhid, nclass, loss_type, dropout=0.5 , mode='SN', d_type='gin', num_mlp_layers=2, num_layers=2):
        super(Discriminator, self).__init__()
        if d_type == 'gin':
            self.dnet = GIN(num_layers, num_mlp_layers, nfeat, nhid, nclass, dropout, False, 'sum', 'sum')
        elif d_type == 'gin_nonorm':
            self.dnet = GIN_nonorm(num_layers, num_mlp_layers, nfeat, nhid, nclass, dropout, False, 'sum', 'sum')
        else:
            self.dnet = GCN(nfeat, nhid, nclass, dropout)
        
        self.dropout = dropout
        self.loss_type = loss_type
        self.d_type = d_type

        if mode == 'SN':
            print('D Use SN')
            self.dnet = _add_sn(self.dnet)

       
    def forward(self, x, adj, train_flag=True):
        if 'gin' in self.d_type:
            # hidden_rep, output = self.dnet(x, adj)
            return  self.dnet(x, adj)
        else:
            return self.dnet(x, adj, train_flag)


    def compute_gan_loss(self, output_real, output_fake):
        r"""
        Computes GAN loss for discriminator.
        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.
        Returns:
            lossD (Tensor): A batch of GAN losses for the discriminator.
        """
        # Compute loss for D
        if self.loss_type == "gan" or self.loss_type == "ns":
            lossD = losses.minimax_loss_dis(output_fake=output_fake,
                                           output_real=output_real)

        elif self.loss_type == "hinge":
            lossD = losses.hinge_loss_dis(output_fake=output_fake,
                                         output_real=output_real)

        elif self.loss_type == "wasserstein":
            lossD = losses.wasserstein_loss_dis(output_fake=output_fake,
                                               output_real=output_real)

        else:
            raise ValueError("Invalid loss_type selected.")

        return lossD

    def compute_acc(self, output, labels):
        r"""
        Computes probabilities from real/fake images logits.
        Args:
            output_real (Tensor): A batch of output logits of shape (N, 1) from real images.
            output_fake (Tensor): A batch of output logits of shape (N, 1) from fake images.
        Returns:
            tuple: Average probabilities of real/fake image considered as real for the batch.
        """
        pred = torch.where(output.sigmoid()>0.5,1.0,0.0)
        correct = torch.sum(pred == labels)
        return correct.item() * 1.0 / len(labels)
