import os,sys

# os.chdir(sys.path[0])
sys.path.append("./")
import torch
import time
import yaml
import math
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import torch.utils.data as Data
from torch.autograd import Variable
from torch_sparse import SparseTensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import os,sys

os.chdir(sys.path[0])
sys.path.append('..')
from modules.losses import compute_D_loss
from utils import *
from utils import _fetch_data
from modules.eval_metric import *
from modules.discriminator import Discriminator
from gnn_model.gin import GIN
from gnn_model.gcn import GCN
from pgd import update_features
sys.path.append('../PGD/')
from attacks.pgd import PGD

def main(args):
    dataset = args.dataset
    suffix = args.suffix
    D_acc = args.D_acc
    gcn_save_file = '../checkpoint/surrogate_model_gcn/' + dataset + '_partition'
    rep_save_file = '../checkpoint/surrogate_model_gin/' + dataset +'_hmlp'
    fig_save_file = 'figures/'+ dataset + '/'
    graph_save_file = 'new_graphs/' + dataset + '/' 
    netD_save_file = 'checkpoint/' + dataset + '/'
    writer = SummaryWriter('tensorboard/' + dataset +  '/' + suffix)
    yaml_path = '../config.yml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = f.read()
        conf_dic = yaml.load(config) 
    if not os.path.exists(fig_save_file):
        os.makedirs(fig_save_file)
    if not os.path.exists(graph_save_file):
        os.makedirs(graph_save_file)
    if not os.path.exists(netD_save_file):
        os.makedirs(netD_save_file)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")


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

    # adj_topo_tensor = torch.tensor(adj.toarray(), dtype=torch.float, device=device)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)

    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    
    feat_lim_max = conf_dic[dataset]['feat_max']
    feat_lim_min = conf_dic[dataset]['feat_min']
    feat_init_max = args.init_scale
    feat_init_min = -args.init_scale
    edge_budget = conf_dic[dataset]['edge_budget']
    # node_budget = conf_dic[dataset]['node_budget']
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    deg = torch.FloatTensor(degree).flatten().to(device)
    feat_num = int(features.sum(1).mean())
    
    # mask = np.arange(labels.shape[0])
    # train_mask, val_mask, test_mask = train_val_test_split_tabular(mask, train_size=0.64, val_size=0.16, test_size=0.2, random_state=seed)
    split = np.aload('../datasets/splits/' + dataset+ '_split.npy').item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']
    index_train = torch.LongTensor(train_mask).to(device)
    index_val = torch.LongTensor(val_mask).to(device)
    index_test = torch.LongTensor(test_mask).to(device)
    stopper = EarlyStop_tdgia_loss(patience=5000)
    # rep_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    rep_net =  GIN(2, 2, features.shape[1], 64, labels.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    
    # Surrogate model
    surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    victim_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    victim_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))

    surro_net.eval()
    victim_net.eval()
    rep_net.eval()
    for p in surro_net.parameters():
        p.requires_grad = False
    for p in victim_net.parameters():
        p.requires_grad = False
    for p in rep_net.parameters():
        p.requires_grad = False
 
    
    logits = victim_net(feat, nor_adj_tensor, train_flag=False)

    logp = F.log_softmax(logits, dim=1)
    acc = accuracy(logp, labels)
    print('Acc:',acc)
    print('Train Acc:',accuracy(logp[train_mask], labels[train_mask]))
    print('Valid Acc:',accuracy(logp[val_mask], labels[val_mask]))
    print('Test Acc:',accuracy(logp[test_mask], labels[test_mask]))
    sec = torch.LongTensor(worst_case_class(logits, labels_np)).to(device)
    
    netD = Discriminator(features.shape[1], 64, 1, args.loss_type, 0, args.D_sn, d_type=args.D_type).to(device)
    optimizer_D = torch.optim.Adam([{'params': netD.parameters()}], lr=args.lr_D, weight_decay=args.wd_D)

    new_row = adj_tensor.coalesce().indices()[0]
    new_col = adj_tensor.coalesce().indices()[1]
    value = adj_tensor.coalesce().values()
    adj_st = SparseTensor(row=new_row, col=new_col, value=value)
    new_adj_tmp = adj_st
    new_feat_tmp = feat
    # Discriminator
    attacker = PGD(epsilon=args.lr, n_epoch=0, n_inject_max=index_test.shape[0],
                 n_edge_max=edge_budget, feat_lim_min=feat_init_min, feat_lim_max=feat_init_max,
                 device=device,  early_stop=1, eval_metric=accuracy)

    new_adj_tensor, inj_feat = attacker.attack(model=surro_net,
                                            adj=adj,
                                            features=feat,
                                            target_idx=index_test,
                                            labels=labels, sec=sec)

    new_adj = sparse_tensor_to_torch_sparse_mx(new_adj_tensor)
    new_n = new_adj_tensor.shape[0]
    new_feat = torch.cat([new_feat_tmp, inj_feat])
    print('inj_feat', inj_feat.shape)
    print('adj_attack', new_adj.shape)
    fake_label = torch.full((new_n-n,1), 0.0, device=device)
    

    Dopt_ep = args.st_epoch

    real_dataset = Data.TensorDataset(torch.LongTensor(test_mask))
    fake_dataset = Data.TensorDataset(torch.LongTensor(np.arange(n,new_n)))
    print('args.batch_size',args.batch_size)
    batch_loader_G = Data.DataLoader(dataset=fake_dataset, batch_size=args.batch_size, num_workers=16,  shuffle=True)
    batch_loader_D_real = Data.DataLoader(dataset=real_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True)
    batch_loader_D_fake = Data.DataLoader(dataset=fake_dataset, batch_size=args.batch_size, num_workers=16,  shuffle=True)
    iter_loader_D_real = iter(batch_loader_D_real)
    iter_loader_D_fake = iter(batch_loader_D_fake)
    iter_loader_G = iter(batch_loader_G)
    if args.st_epoch != 0:
        Dopt_ep = 2579
        netD.load_state_dict(torch.load(netD_save_file +'netD_' + suffix + '_checkpoint.pt'))
        new_adj, new_features, labels_np = load_npz(graph_save_file + suffix  + '.npz')
        new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj)
        new_feat = torch.from_numpy(new_features.toarray().astype('double')).float().to(device)
    start = time.time()
    for epoch in range(args.st_epoch, args.nepochs):
        netD.train()
        for ep in range(args.Dopt):
            if 'k' not in dataset:
            # if True:
                iter_loader_D_real, real_batch = _fetch_data(iter_dataloader=iter_loader_D_real, dataloader=batch_loader_D_real)
                iter_loader_D_fake, fake_batch = _fetch_data(iter_dataloader=iter_loader_D_fake, dataloader=batch_loader_D_fake)
                real_batch = real_batch[0]
                fake_batch = fake_batch[0]
            else:
                fake_batch = np.arange(n,new_n)
                real_batch =  np.random.randint(0, adj.shape[0], (len(fake_batch)))
            train_loss_D, train_acc_D, train_acc_real, train_acc_fake = compute_D_loss(real_batch, fake_batch, netD, adj_tensor, feat, new_feat, new_adj_tensor,args.clipfeat)
            optimizer_D.zero_grad()
            train_loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.1)
            optimizer_D.step()
            train_loss_D_detach = train_loss_D.detach().item()
            del train_loss_D
            # writer.add_scalar('Each opt D/acc real D', train_acc_real, Dopt_ep)
            # writer.add_scalar('Each opt D/acc fake D', train_acc_fake, Dopt_ep)
            # # writer.add_scalar('Each opt D/acc fake train D', acc_fake_train_D, Dopt_ep)
            # writer.add_scalar('Each opt D/acc D', train_acc_D, Dopt_ep)
            # writer.add_scalar('Each opt D/loss D', train_loss_D_detach, Dopt_ep)
            Dopt_ep += 1

        netD.eval()
        # print('Epoch',epoch)
        # if train_acc_D > 0.95:
        #     n_ep = 2
        #     if train_acc_D > 0.98:
        #         n_ep = 5
        if 'k' not in dataset:
        # if True:
            inj_feat, loss_G_atk, loss_G_fake, train_loss_G, matk, acc_fake_G = update_features(attacker, surro_net, new_adj_tensor, feat, inj_feat, labels, sec, index_test, netD=netD, rep_net=rep_net, loss_type=args.loss_type, atk_alpha=args.atk_alpha, alpha=args.alpha, beta=args.beta, batch_loader_G=batch_loader_G, iter_loader_G=iter_loader_G)
        else:
            inj_feat, loss_G_atk, loss_G_fake, train_loss_G, matk, acc_fake_G = update_features(attacker, surro_net, new_adj_tensor, feat, inj_feat, labels, sec, index_test, netD=netD, rep_net=rep_net, loss_type=args.loss_type, atk_alpha=args.atk_alpha, alpha=args.alpha, beta=args.beta, fake_batch=fake_batch)
        
        inj_feat = torch.clamp(inj_feat, feat_lim_min, feat_lim_max)
        new_feat = torch.cat((feat, inj_feat),0)
        
        # writer.add_scalar('Metric/In D acc',acc_fake_G, epoch)
        # val_GFD = calculate_graphfd(rep_net, adj_tensor, feat, new_adj_tensor, new_feat, np.arange(n), np.arange(n, new_feat.shape[0]))
        # print('D acc:', train_acc_D)
        # print('D loss:', train_loss_D_detach)
        # print('G attack loss:', loss_G_atk)
        # print('G consistency loss:', loss_G_fake.item())
        # # print('G diversity loss:', loss_G_div)
        # print('G loss:', train_loss_G.item())
        # print('Metric: Attack Success', matk)
        # print('Metric: Consistency GFD', val_GFD)

        # writer.add_scalar('Discriminator/acc D', train_acc_D, epoch)
        # writer.add_scalar('Discriminator/loss D', train_loss_D_detach, epoch)

        # writer.add_scalar('loss_G/Consistency', loss_G_fake, epoch)
        # # writer.add_scalar('loss_G/Diversity',loss_G_div, epoch)
        # writer.add_scalar('loss_G/Attack',loss_G_atk, epoch)
        # writer.add_scalar('loss_G/Overall',train_loss_G, epoch)

        # writer.add_scalar('Metric/Consist GFD', val_GFD, epoch)
        # writer.add_scalar('Metric/Attack Success', matk, epoch)

        # stopper.save_checkpoint(netD, netD_save_file + 'netD_'+suffix)
        # stopper.save_graph(new_adj, new_feat, labels_np, graph_save_file+suffix)
        # if epoch % 5 == 0 and epoch > 70:
        #     new_adj_sp = sp.csr_matrix(new_adj)
        #     new_feature_sp = sp.csr_matrix(new_feat.detach().cpu().numpy())
        #     np.savez(graph_save_file+suffix + '_' + str(epoch) + '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
        #         adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
        #         attr_shape=new_feature_sp.shape, labels=labels_np)
        
    surro_net.eval()
    new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
    new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
    new_feature_np = new_feat.detach().cpu().numpy()
    new_n = new_feat.shape[0]
    
    labels = torch.from_numpy(labels_np).cuda()
    new_output = surro_net(new_feat, new_nor_adj_tensor)
    atk_suc =((labels[test_mask] != new_output[test_mask].argmax(1)).sum()).item()/test_mask.shape[0]
    print('atk_suc',atk_suc)
    print('New Graph:')
    new_adj_sp = sp.csr_matrix(new_adj)
    new_feature_sp = sp.csr_matrix(new_feature_np)
    dur = time.time() - start
    print('during', dur,'s; ', dur/60, 'm; ', dur/3600, 'h')
    np.savez(graph_save_file+suffix + '.npz', adj_data=new_adj_sp.data, adj_indices=new_adj_sp.indices, adj_indptr=new_adj_sp.indptr,
        adj_shape=new_adj_sp.shape, attr_data=new_feature_sp.data, attr_indices=new_feature_sp.indices, attr_indptr=new_feature_sp.indptr,
        attr_shape=new_feature_sp.shape, labels=labels_np)

    evaluation(new_feat, new_adj_tensor, new_adj, feat, adj_tensor, adj, rep_net, n, fig_save_file, suffix)
    

    print('*'*15)
    return

if __name__ == '__main__':
    setup_seed(123)

    parser = argparse.ArgumentParser(description="PGD")
    parser.add_argument('-f')
    parser.add_argument('--dataset',default='ogbproducts')
    parser.add_argument('--attrepoch',type=int,default=10000000)
    parser.add_argument('--models',nargs='+',default=['gcn_lm'])
    parser.add_argument('--modelsextra',nargs='+',default=[])
    parser.add_argument('--modelseval',nargs='+',default=['gcn_lm','graphsage_norm','sgcn','rgcn','tagcn','appnp','gin'])
    #extra models are only used for label approximation.
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--strategy',default='gia')
    parser.add_argument('--test_rate',type=int,default=0)
    parser.add_argument('--test',type=int,default=50000)
    parser.add_argument('--lr',type=float,default=1)
    parser.add_argument('--step',type=float,default=0.2)
    parser.add_argument('--weight1',type=float,default=0.9)
    parser.add_argument('--weight2',type=float,default=0.1)
    parser.add_argument('--add_rate',type=float,default=1)
    parser.add_argument('--init_scale',type=float,default=20)
    parser.add_argument('--opt',default="sin")
    parser.add_argument('--add_num',type=float,default=500)
    parser.add_argument('--max_connections',type=int,default=4)
    parser.add_argument('--connections_self',type=int,default=0)

    parser.add_argument('--apply_norm',type=int,default=1)
    #also evaluate on surrogate models
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--suffix',type=str,default='debug')
    parser.add_argument('--surro_type', default='gcn',help='surrogate gnn model')
    parser.add_argument('--nepochs', type=int, default=100000, help='number of epochs')
    parser.add_argument('--st_epoch', type=int, default=0, help='number of epochs')
    parser.add_argument('--lr_D', default=1e-3, type=float, help='learning rate of Discriminator')
    parser.add_argument('--Dopt', type=int, default=10000, help='Discriminator optimize Dopt times, G optimize 1 time.')
    parser.add_argument('--D_sn', type=str, default='none', choices=['none', 'SN'] , help='whether Discriminator use spectral_norm')
    parser.add_argument('--D_type', type=str, default='gin_nonorm',  help='Discriminator type')
    parser.add_argument('--D_acc', type=float, default=0.8, help='Appropriate Discriminator accuracy')
    parser.add_argument('--Gopt', type=int, default=1, help='Inject feature with the feature of last epoch')
    parser.add_argument('--loss_type', type=str, default='gan', choices=['gan', 'ns', 'hinge', 'wasserstein'], help='Loss type.')
    parser.add_argument('--step_size', type=int, default=100, help='Step size of the optimizer scheduler')
    parser.add_argument('--patience', type=int, default=500, help='Patience of feature optimization')
    
    parser.add_argument('--atk_alpha', type=float, default=0, help='the coefficient of GAN loss in G loss')
    parser.add_argument('--alpha', type=float, default=1, help='the coefficient of GAN loss in G loss')
    parser.add_argument('--beta', type=float, default=0, help='the coefficient of diversity loss in G loss')
    parser.add_argument('--clipfeat', type=bool, default=False, help='the coefficient of GAN when choosing edges to connect')      
    parser.add_argument('--wd_D', type=float, default=0.1)  
    parser.add_argument('--div_bs', type=int, default=64)      

    args=parser.parse_args()
    opts = args.__dict__.copy()
    print(opts)
    att_sucess = main(args) 


'''
CUDA_VISIBLE_DEVICES=3 nohup python -u run_imppgd.py  --dataset ogbproducts --batch_size 2099 --D_type gin --suffix pgd_cana_time1 --wd_D 0.1 --atk_alpha 1 --alpha 10 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 1 --nepochs 800 > log/ogbproducts/pgd_cana_time.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u run_imppgd.py  --dataset reddit --D_type gin --suffix pgd_cana_time --wd_D 0.1 --atk_alpha 1 --alpha 10 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 1 --nepochs 700 > log/reddit/pgd_cana_time.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u run_imppgd.py  --dataset ogbarxiv --batch_size 1800 --D_type gin --suffix pgd_cana_time --wd_D 0.1 --atk_alpha 1 --alpha 10 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 1 --nepochs 1500 > log/ogbarxiv/pgd_cana_time.log 2>&1 &
'''

'''
CUDA_VISIBLE_DEVICES=5 nohup python -u run_imppgd.py  --dataset ogbproducts --batch_size 2099 --D_type gin --suffix alpha10_bs2099 --wd_D 0.1 --atk_alpha 1 --alpha 10 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 1 --nepochs 1000 > log/ogbproducts/alpha10_bs2099.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u run_imppgd.py  --dataset reddit --st_epoch 128 --D_type gin --suffix alpha20_Dopt20 --init_scale 5 --wd_D 0.1 --atk_alpha 1 --alpha 20 --beta 0.01 --lr 1e-1 --lr_D 1e-2 --Dopt 20 --nepochs 20000 >> log/reddit/alpha20_Dopt20.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python -u run_imppgd.py  --dataset ogbarxiv --batch_size 1800 --D_type gin --suffix alpha50_Dopt4 --wd_D 0.1 --init_scale 0.5 --atk_alpha 1 --alpha 50 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 4 --nepochs 20000 > log/ogbarxiv/alpha50_Dopt4.log 2>&1 &


30inj
CUDA_VISIBLE_DEVICES=2 nohup python -u run_imppgd.py  --dataset ogbproducts --batch_size 2000 --D_type gin --suffix 30inj_alpha10_bs2099_1e-2 --wd_D 0.1 --atk_alpha 1 --alpha 10 --beta 0.01 --lr 1e-2 --lr_D 1e-3 --Dopt 1 --nepochs 1000 > log/ogbproducts/30inj_alpha10_bs2099_1e-2.log 2>&1 &


'''
