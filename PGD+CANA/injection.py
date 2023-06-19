
from operator import add
import random
# from attacks import utils
import torch.nn.functional as F
import numpy as np
import torch
import scipy.sparse as sp
import sys
sys.path.append('..')
from modules.losses import compute_gan_loss, percept_loss

def init_feat(num, features, device, style="sample", feat_lim_min=-1, feat_lim_max=1):
    if style.lower() == "sample":
        # do random sample from features to init x
        feat_len = features.size(0)
        x = torch.empty((num,features.size(1)),device=features.device)
        sel_idx = torch.randint(0,feat_len,(num,1))
        x = features[sel_idx.view(-1)].clone()
    elif style.lower() == "normal":
        x = torch.randn((num,features.size(1))).to(features.device)
    elif style.lower() == "zeros":
        x = torch.zeros((num,features.size(1))).to(features.device)
    else:
        x = np.random.normal(loc=0, scale=feat_lim_max/10, size=(num, features.size(1)))
        x = torch.FloatTensor(x).to(device)
        # x = utils.feat_preprocess(features=x, device=device)
    return x

# pgd feature upd
def update_features(attacker, model, adj_attack, features, features_attack, origin_labels, target_idx, n_epoch=499, dist="cos",netD=None, rep_net=None):
    attacker.early_stop.reset()
    if hasattr(attacker, 'disguise_coe'):
        disguise_coe = attacker.disguise_coe
    else:
        disguise_coe = 0

    epsilon = attacker.epsilon
    print(n_epoch)
    print(attacker.n_epoch)
    n_epoch = min(n_epoch,attacker.n_epoch)
    feat_lim_min, feat_lim_max = attacker.feat_lim_min, attacker.feat_lim_max
    n_total = features.shape[0]
    if dist.lower()=="cos":
        dis = lambda x: F.cosine_similarity(x[0],x[1])
    elif dist.lower() == "l2":
        dis = lambda x: F.pairwise_distance(x[0],x[1],p=2)
    else:
        raise Exception(f"no implementation for {dist}")
    
    # features_attack = utils.feat_preprocess(features=features_attack, device=attacker.device)
    model.eval()
    # initialize the features with averaged neighbors' features
    # with torch.no_grad():
    #     features_attack = adj_attack @ torch.cat((features,features_attack),dim=0) 
    #     features_attack = features_attack[n_total:]
    attack_degs = torch.unique(adj_attack.coo()[1],return_counts=True)[1][-features_attack.size(0):]
    for i in range(n_epoch):
        features_attack.requires_grad_(True)
        features_attack.retain_grad()
        features_concat = torch.cat((features, features_attack), dim=0)
        # features_concat = feat_normalize(features_concat,norm='arctan')
        # features_concat = (features_concat - features_concat.mean()) / features_concat.std()
        # features_concat = torch.arctan(features_concat) / pi2
        pred = model(features_concat, adj_attack)
        # stablize the pred_loss, only if disguise_coe > 0
        weights = pred[target_idx,origin_labels[target_idx]].exp()>=min(disguise_coe,1e-8)
        pred_loss = attacker.loss(pred[:n_total][target_idx],
                                   origin_labels[target_idx],reduction='none')
        pred_loss = (pred_loss*weights).mean()
        # # shall be pre_loss = +loss
        # pred_loss = attacker.loss(pred[:n_total][target_idx],
        #                        origin_labels[target_idx]).to(attacker.device)
        # (inversely) maximize the differences 
        # between attacked features and neighbors' features
        with torch.no_grad():
            features_propagate = adj_attack @ torch.cat((features,torch.zeros(features_attack.size()).to(features.device)),dim=0) 
            features_propagate = features_propagate[n_total:]/attack_degs.unsqueeze(1)
        
        
        all_emb_list = rep_net(features_concat, adj_attack)
        pred_fake_G = netD(features_concat, adj_attack)[1]
        loss_G_fake = compute_gan_loss(attacker.loss_type,pred_fake_G[n_total:])
        loss_G_div = percept_loss(all_emb_list, pred_fake_G.shape[0]-n_total)
        loss = pred_loss +  attacker.alpha * loss_G_fake + attacker.beta * loss_G_div 
        model.zero_grad()
        loss.backward()
        grad = features_attack.grad.data
        features_attack = features_attack.clone() + epsilon * grad.sign()
        # features_attack = torch.clamp(features_attack, feat_lim_min, feat_lim_max)
        features_attack = features_attack.detach()
        test_score = attacker.eval_metric(pred[:n_total][target_idx],
                                      origin_labels[target_idx])
        if attacker.early_stop:
            attacker.early_stop(test_score)
            if attacker.early_stop.stop:
                print("Attacking: Early stopped.")
                attacker.early_stop.reset()
                return features_attack
        if attacker.verbose:
            print(
                "Attacking: Epoch {}, Loss: {:.5f},  PredLoss: {:.5f}, ConsistLoss: {:.5f}, Surrogate test score: {:.5f}".format(i, loss, pred_loss, loss_G_fake, test_score),
                end='\r' if i != n_epoch - 1 else '\n')
    return features_attack



from torch_sparse import SparseTensor


def apgd_injection(attacker, model, adj, n_inject, n_edge_max, features, features_attack, target_idx, origin_labels, 
                    device, a_epoch, optim="adam", old_reg=False, real_target_idx=None, homophily=None,netD=None, rep_net=None):
    # torch.autograd.set_detect_anomaly(True)
    model.to(attacker.device)
    model.eval()
    n_epoch = attacker.n_epoch
    n_total = features.size(0)
    device = attacker.device
    epsilon = attacker.epsilon
    # setup the edge entries for optimization
    new_x = torch.cat([torch.LongTensor([i+n_total]).repeat(target_idx.size(0))
                      for i in range(n_inject)]).to(device)
    new_y = target_idx.repeat(n_inject).to(device)
    assert new_x.size() == new_y.size()
    vals = torch.zeros(new_x.size(0)).to(device)
    print(f"#original edges {adj.nnz()}, #target idx {len(target_idx)}, #init edges {vals.size(0)}")
    # jointly update adjecency matrix & features
    if adj.size(0)>n_total:
        print("init edge weights from the previous results")
        # that's a attacked adj
        orig_adj = adj[:n_total,:n_total]
        x, y, z = orig_adj.coo()
        # now we init val with the attacked graph
        vals[:] = 0
        x_inj, y_inj, _ = adj[n_total:,:].coo()
        idx_map = {}
        for (i, idx) in enumerate(target_idx):
            idx_map[idx.item()] = i 
        for i in range(n_inject*n_edge_max):
            xx, yy = x_inj[i], y_inj[i]
            pos = xx*len(target_idx)+idx_map[yy.cpu().item()]
            vals[pos] = 1
        old_vals = vals.clone()
        if old_reg:
            print("upd vals via layerwise gradient")
            # if attacker.n_epoch%10 == 0:
            #     vals[:] = 0
            # elif attacker.n_epoch%10 == 1:
            #     vals[:] = 1
            # else:
            # per_idx = torch.randperm(vals.size(0))
            # vals[:] = vals[per_idx]
            # vals[:] = 1
    else:
        old_vals = None
        x, y, z = adj.coo()
    
    z = torch.ones(x.size(0)).to(device) if z == None else z
    isolate_idx = torch.nonzero((adj.sum(-1)==0)[:n_total].long(),as_tuple=True)[0].cpu()
    # print(len(isolate_idx))
    # print((adj.sum(-1)==0)[:n_total].sum())
    makeup_x = []
    makeup_y = []
    makeup_z = []
    for iidx in isolate_idx:
        # print(iidx in target_idx)
        makeup_x.append(iidx)
        makeup_y.append(iidx)
        makeup_z.append(1)
    x = torch.cat((x,torch.LongTensor(makeup_x).to(device)),dim=0)
    y = torch.cat((y,torch.LongTensor(makeup_y).to(device)),dim=0)
    z = torch.cat((z,torch.LongTensor(makeup_z).to(device)),dim=0)
    print(f"add self-con for {len(isolate_idx)} nodes")
    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    vals.requires_grad_(True)
    vals.retain_grad()
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))
    
    if optim == "adam":
        optimizer_adj = torch.optim.Adam([vals],epsilon)
    features_concat = torch.cat((features, features_attack), dim=0)
    old_layer_output = None
    orig_layer_output = None
    # torch.autograd.set_detect_anomaly(True) 
    # if old_reg:
    #     n_epoch//=2
    real_target_idx = target_idx[target_idx<origin_labels.size(0)] if real_target_idx==None else real_target_idx
    print(f"wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww: {real_target_idx.size()}")
    
    beta = 0.01 if n_edge_max >= 100 else 1
    for i in range(a_epoch):
        if old_reg:
            # vals = torch.clamp(vals.detach(),0,1)
            # vals.requires_grad_(True)
            # vals.retain_grad()
            # adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))
            layer_pred = model(features_concat, adj_attack, layers=2)
            with torch.no_grad():
                if old_layer_output==None:
                    old_layer_output = model(features_concat,adj,layers=2)
                    orig_layer_output = model(features_concat,adj[:n_total,:n_total],layers=2)
            pred = model.con_forward(layer_pred, adj_attack, layers=3)
        else:
            pred = model(features_concat, adj_attack)
        # stablize the pred_loss, only if disguise_coe > 0
        # weights = pred[real_target_idx,origin_labels[real_target_idx]].exp()>=min(attacker.disguise_coe,1e-8)
        # pred_loss = attacker.loss(pred[:n_total][real_target_idx],
        #                         origin_labels[real_target_idx],reduction='none')
        # pred_loss = (pred_loss*weights).mean()
        
        
        pred_loss = attacker.loss(pred[:n_total][real_target_idx],
                              origin_labels[real_target_idx])
        
        # sparsity loss for the adjacency matrix, based on L1 norm
        # if optim=="adam" and not model.use_ln:
        if optim=="adam":
            # it seems a more relax sparsity regularizer would be better
            # so the effects of APGD is more like pruning useless edges
            sparsity_loss = beta*(n_edge_max*n_inject-torch.norm(vals,p=1))
            # sparsity_loss = (n_edge_max-vals.view(n_inject,-1).sum(-1)).mean()
            # sparsity_loss = -0.01*torch.norm(vals,p=1)
            # sparsity_loss = -0.1*abs(vals.view(n_inject,-1).sum(-1)-n_edge_max).mean()
        else:
            # pgd upd seems work better when LN is used inner layers
            sparsity_loss = -0.01*abs(vals.view(n_inject,-1).sum(-1)-n_edge_max).mean()
        # print(f"{pred_loss}, {loss_G_fake}, {loss_G_div}")
        pred_loss = -pred_loss-sparsity_loss
        if old_reg:
            #ILAF
            # pred_loss += -(torch.norm(layer_pred)/torch.norm(old_layer_output)).mean()-F.cosine_similarity(layer_pred,old_layer_output).mean()#+0.99*sparsity_loss
            #ILAP
            ilap_loss = -(layer_pred[real_target_idx]-orig_layer_output[real_target_idx])*(old_layer_output[real_target_idx]-orig_layer_output[real_target_idx])
            pred_loss += ilap_loss.sum(-1).mean()
            # print(pred_loss.size())
        # print(f"edge weights min: {vals.min()}, max: {vals.max()}")
        all_emb_list = rep_net(features_concat, adj_attack)
        pred_fake_G = netD(features_concat, adj_attack)[1]
        loss_G_fake = compute_gan_loss(attacker.loss_type,pred_fake_G[n_total:])
        loss_G_div = percept_loss(all_emb_list, pred_fake_G.shape[0]-n_total)
        loss = pred_loss +  attacker.alpha * loss_G_fake + attacker.beta * loss_G_div 
        if optim == "adam":
            optimizer_adj.zero_grad()
        loss.backward(retain_graph=True)
        # print(f"vals.isnan() {vals.isnan().sum()}, vals.grad.data.isnan() {vals.grad.data.isnan().sum()}")
        if optim == "adam":
            optimizer_adj.step()
            # vals= F.sigmoid(vals)
        else:
            # this version
            grad = vals.grad.data
            vals = vals.detach() - epsilon * grad.sign()
            # vals = torch.clamp(vals,0,1)
            vals.requires_grad_(True)
            vals.retain_grad()
            adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.cat((z, vals, vals), dim=0))


        test_score = attacker.eval_metric(pred[:n_total][real_target_idx],
                                      origin_labels[real_target_idx])
        if attacker.verbose:
            print("Attacking Edges: Epoch {}, Loss: {:.5f},  PredLoss: {:.5f}, ConsistLoss: {:.5f}, Surrogate test score: {:.2f}, injected edge {:}".format(
                    i, loss, pred_loss, loss_G_fake, test_score, vals[:len(target_idx)].sum()),end='\r' if i != n_epoch - 1 else '\n')


    # select edges with higher weights as the final injection matrix
    tmp_vals = -vals.detach().view(n_inject, -1)
    sel_idx = tmp_vals.argsort(dim=-1)[:, :n_edge_max]
    sel_mask = torch.zeros(tmp_vals.size()).bool()
    for i in range(sel_idx.size(0)):
        sel_mask[i, sel_idx[i]] = True
    sel_idx = torch.nonzero(sel_mask.view(-1)).squeeze()
    # sel_idx = torch.nonzero(torch.logical_and(sel_mask.view(-1).to(vals.device),vals>0.1)).squeeze()
    # print(sel_idx) 
    new_x = new_x[sel_idx]
    new_y = new_y[sel_idx]
    print(f"Finally injected edges {len(new_x)}, minimum vals {vals[sel_idx].min()}, maximum vals {vals[sel_idx].max()}")
    # x,y,_ = adj.coo()
    new_row = torch.cat((x, new_x, new_y), dim=0)
    new_col = torch.cat((y, new_y, new_x), dim=0)
    adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=device))
    if old_vals!=None:
        new_vals = torch.zeros(old_vals.size()).to(old_vals.device)
        new_vals[sel_idx] = 1
        print(f"number of modifications: {(old_vals-new_vals).abs().sum()}")
        print(f"added: {((-old_vals+new_vals)>0).sum()}")
        print(f"removed: {((old_vals-new_vals)>0).sum()}")
    return adj_attack



def node_sim_estimate(x, adj, num, style='sample'):
    """
    estimate the mean and variance from the observed data points
    """
    sims = node_sim_analysis(adj,x)
    if style.lower() == 'random':
        hs = np.random.choice(sims,size=(num,))
        hs = torch.FloatTensor(hs).to(x.device)
    else:
        # mean, var = sims.mean(), sims.var()
        # hs = torch.randn((num,)).to(x.device)
        # hs = mean + hs*torch.pow(torch.tensor(var),0.5)
        from scipy.stats import skewnorm
        a, loc, scale = skewnorm.fit(sims)
        hs = skewnorm(a, loc, scale).rvs(num)
        hs = torch.FloatTensor(hs).to(x.device)
    return hs

def node_sim_analysis(adj, x):
    print('node_sim_analysis adj',adj)
    adj = gcn_norm(adj,add_self_loops=False)
    x_neg = adj @ x
    node_sims = F.cosine_similarity(x_neg,x).cpu().numpy()
    return node_sims

from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1., dtype=None)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1) + 1e-4
    deg_inv_sqrt = deg.pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t
