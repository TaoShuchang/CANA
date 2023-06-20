
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from modules.losses import compute_gan_loss,percept_loss
from modules.eval_metric import *
from utils import _fetch_data
from utils import *

import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from utils import sparse_mx_to_torch_sparse_tensor


def eval_acc(pred, labels, mask=None):

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0

    acc = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

    return acc


class PGD(object):
    r"""

    Description
    -----------
    Graph injection attack version of Projected Gradient Descent attack (`PGD <https://arxiv.org/abs/1706.06083>`__).

    Parameters
    ----------
    epsilon : float
        Perturbation level on features.
    n_epoch : int
        Epoch of perturbations.
    n_inject_max : int
        Maximum number of injected nodes.
    n_edge_max : int
        Maximum number of edges of injected nodes.
    feat_lim_min : float
        Minimum limit of features.
    feat_lim_max : float
        Maximum limit of features.
    loss : func of torch.nn.functional, optional
        Loss function compatible with ``torch.nn.functional``. Default: ``F.nll_loss``.
    eval_metric : func of grb.evaluator.metric, optional
        Evaluation metric. Default: ``eval_acc``.
    device : str, optional
        Device used to host data. Default: ``cpu``.
    early_stop : bool, optional
        Whether to early stop. Default: ``False``.
    verbose : bool, optional
        Whether to display logs. Default: ``True``.

    """

    def __init__(self,
                 epsilon,
                 n_epoch,
                 n_inject_max,
                 n_edge_max,
                 feat_lim_min,
                 feat_lim_max,
                 loss=F.nll_loss,
                 eval_metric=eval_acc,
                 device='cpu',
                 early_stop=False,
                 verbose=True,
                 disguise_coe=0):
        self.device = device
        self.epsilon = epsilon
        self.n_epoch = n_epoch
        self.n_inject_max = n_inject_max
        self.n_edge_max = n_edge_max
        self.feat_lim_min = feat_lim_min
        self.feat_lim_max = feat_lim_max
        self.loss = loss
        self.eval_metric = eval_metric
        self.verbose = verbose
        self.disguise_coe = disguise_coe
        # Early stop
        if early_stop:
            self.early_stop = EarlyStop(patience=early_stop, epsilon=1e-4)
        else:
            self.early_stop = early_stop

    def attack(self, model, adj, features, target_idx, labels=None, sec=None):
        model.to(self.device)
        model.eval()
        n_total, n_feat = features.shape

        origin_labels = labels.view(-1)
        
        new_adj = self.injection(adj=adj,
                                    n_inject=self.n_inject_max,
                                    n_node=n_total,
                                    target_idx=target_idx)
        new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(target_idx.device)

        # Random initialization
        inj_features = np.random.normal(loc=0, scale=self.feat_lim_max / 10,
                                           size=(self.n_inject_max, n_feat))
        
        inj_feat = torch.from_numpy(inj_features.astype('double')).float().to(target_idx.device)

        return new_adj_tensor, inj_feat

    def injection(self, adj, n_inject, n_node, target_idx):
        r"""

        Description
        -----------
        Randomly inject nodes to target nodes.

        Parameters
        ----------
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        n_inject : int
            Number of injection.
        n_node : int
            Number of all nodes.
        target_idx : torch.Tensor
            Mask of attack target nodes in form of ``N * 1`` torch bool tensor.

        Returns
        -------
        adj_attack : scipy.sparse.csr.csr_matrix
            Adversarial adjacency matrix in form of :math:`(N + N_{inject})\times(N + N_{inject})` sparse matrix.

        """

        # test_index = torch.where(target_idx)[0]
        target_idx = target_idx.cpu()
        n_test = target_idx.shape[0]
        new_edges_x = []
        new_edges_y = []
        new_data = []
        for i in range(n_inject):
            islinked = np.zeros(n_test)
            for j in range(self.n_edge_max):
                x = i + n_node

                yy = random.randint(0, n_test - 1)
                while islinked[yy] > 0:
                    yy = random.randint(0, n_test - 1)

                islinked[yy] = 1
                y = target_idx[yy]
                new_edges_x.extend([x, y])
                new_edges_y.extend([y, x])
                new_data.extend([1, 1])

        add1 = sp.csr_matrix((n_inject, n_node))
        add2 = sp.csr_matrix((n_node + n_inject, n_inject))
        adj_attack = sp.vstack([adj, add1])
        adj_attack = sp.hstack([adj_attack, add2])
        adj_attack.row = np.hstack([adj_attack.row, new_edges_x])
        adj_attack.col = np.hstack([adj_attack.col, new_edges_y])
        adj_attack.data = np.hstack([adj_attack.data, new_data])

        return adj_attack



# pgd feature upd
def update_features(attacker, model, new_adj_tensor, feat, inj_feat, labels, sec, target_idx, netD=None, rep_net=None, loss_type=None, alpha=None, beta=None, fake_batch=None, batch_loader_G=None, iter_loader_G=None):
    n = feat.shape[0]
    new_n = new_adj_tensor.shape[0]
    ori_label = labels[target_idx]
    best_wrong_label = sec[target_idx]
    inj_feat.requires_grad_(True)
    # inj_feat.retain_grad()
    optimizer=torch.optim.Adam([{'params':[inj_feat]}], lr=attacker.epsilon)
    if fake_batch is None:
        iter_loader_G, fake_batch_G = _fetch_data(iter_dataloader=iter_loader_G, dataloader=batch_loader_G)
        fake_batch = fake_batch_G[0].to(feat.device)
    feat_concat = torch.cat((feat, inj_feat), dim=0)
    # row, column, value = new_adj_tensor.coo()
    pred = model(feat_concat, normalize_tensor(new_adj_tensor))
    # stablize the pred_loss, only if disguise_coe > 0
    loss_G_atk = F.relu(pred[target_idx, ori_label] - pred[target_idx, best_wrong_label]).mean()
    all_emb_list = rep_net(feat_concat, new_adj_tensor)
    pred_fake_G = netD(feat_concat, new_adj_tensor)[1]
    # print('fake_batch',fake_batch.shape)
    loss_G_fake = compute_gan_loss(loss_type,pred_fake_G[fake_batch])
    loss_G_div = percept_loss(all_emb_list, fake_batch)
    all_loss = loss_G_atk + alpha * loss_G_fake + beta * loss_G_div  
    # all_loss = atk_alpha * loss_G_atk + alpha * loss_G_fake 
    fake_label = torch.full((fake_batch.shape[0],1), 0.0, device=pred_fake_G.device)
    acc_fake = netD.compute_acc(pred_fake_G[fake_batch], fake_label )
    metric_atk_suc = ((labels[target_idx] != pred[target_idx].argmax(1)).sum()).item()/len(target_idx)

    optimizer.zero_grad()
    all_loss.backward()
    optimizer.step()
    inj_feat = inj_feat.detach()
    

    return inj_feat, loss_G_atk, loss_G_fake, all_loss, metric_atk_suc, acc_fake

class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """
    def __init__(self, patience=100, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack acore.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True

    def reset(self):
        self.min_score = None
        self.stop = False
        self.count = 0
