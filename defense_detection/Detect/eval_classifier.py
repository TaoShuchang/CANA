
import time
import sys
import os
import csv
import torch
import math
import argparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from pyod.models.iforest import *
from pyod.models.hbos import *
from pyod.models.mcd import *
from pyod.models.knn import *
from pyod.models.cblof import *
from pyod.models.pca import *
from pyod.models.ocsvm import *
from pyod.models.copod import *
from pyod.models.auto_encoder import *
from pyod.models.gmm import *

# from models_gcn import *
# from gcn import *
import os,sys

os.chdir(sys.path[0])
sys.path.append('../..')
from utils import *
from modules.eval_metric import *
from modules.discriminator import Discriminator
from gnn_model.gcn import GCN
from gnn_model.gin import GIN

def evaluate_atk(model,emb, adj, features, labels, surro_net, index_test, device, ninj=0):
    n = emb.shape[0] - ninj
    
    ground_truth = np.ones(n+ninj)
    ground_truth[n:] = 0
    
    # model.fit(emb)
    out = model.fit_predict(emb)
    row_0 = np.where(out == ground_truth[0])[0]
    row_1 = np.where(out == ground_truth[-1])[0]
    if row_0.shape[0] < row_1.shape[0]:
        out = 1 - out
    acc_fake = (ground_truth[n:]==out[n:]).sum()/len(out[n:])
    acc_all = (ground_truth==out).sum()/len(out)
    atk_suc = mask_evaluate(adj, features, labels, surro_net, out, index_test, device)
    print("Test misclassification {:.4%}".format(atk_suc))
    print("Test acc fake {:.4%}".format(acc_fake))
    print("Test acc all {:.4%}".format(acc_all))


def obtain_acc(labels, out, ninj=None):
    # final_pred = torch.where(out.sigmoid()>0.5,1.0,0.0)
    # print('out.sigmoid()',out.sigmoid())
    final_pred = torch.tensor(out).to(labels.device)
    correct = torch.sum(final_pred == labels)
    if ninj is not None:
        correct_fake = torch.sum(final_pred[-ninj:] == labels[-ninj:])

        return final_pred, correct.item() * 1.0 / len(labels), correct_fake.item() * 1.0 / len(labels[-ninj:])
    else:
        return final_pred, correct.item() * 1.0 / len(labels)


def mask_evaluate(new_adj, new_feature, labels, surro_net, pred, index_test,device):
    n = new_adj.shape[0]
    trust_mask = [pred == 1][0].squeeze()
    trust_mask[index_test] = True
    mask_new_feature = new_feature[trust_mask]
    mask_new_feat = torch.FloatTensor(mask_new_feature.toarray()).to(device)
    mask_new_adj = new_adj[trust_mask][:,trust_mask]
    mask_new_adj_tensor =  sparse_mx_to_torch_sparse_tensor(mask_new_adj).to(device)
    mask_new_nor_adj_tensor = normalize_tensor(mask_new_adj_tensor)
    
    mask_output = surro_net(mask_new_feat, mask_new_nor_adj_tensor)
    sum_mask = []
    cur_sum = -1
    for i in range(n):
        cur_sum += trust_mask[i]
        sum_mask.append(cur_sum)
    sum_mask = np.array(sum_mask)
    atk_suc =((labels[index_test] != mask_output[sum_mask[index_test]].argmax(1)).sum()).item()/index_test.shape[0]
    return atk_suc



def main(args):
    gcn_save_file = '../../checkpoint/surrogate_model_gcn/' + args.dataset + '_partition'
    rep_save_file = '../../checkpoint/surrogate_model_gin/' + args.dataset +'_hmlp'
    # graphpath = '../../final_graphs/' + args.dataset + '/' 
    # if 'gnia' in args.suffix:
    #     graphpath = '../../GNIA/new_graphs/' + args.dataset + '/' 
    # elif 'tdgia' in args.suffix:
    #     graphpath = '../../TDGIA/new_graphs/' + args.dataset + '/' 
    # else:
    #     graphpath = '../../PGD/new_graphs/' + args.dataset + '/' 
    if 'gnia' in args.suffix:
        graphpath = '../../GNIA_DINA/new_graphs/' + args.dataset + '/new/' 
    elif 'pgd' in args.suffix:
        graphpath = '../../PGD_DINA/new_graphs/' + args.dataset + '/candidate/' 
    # graphpath = '../../TDGIA_DINA/new_graphs/prestruc/' + args.dataset + '/' 
    graphpath = 'used_gfd/' 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('dataset',args.dataset)
    adj, features, labels_np = load_npz(f'../../datasets/{args.dataset}.npz')
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
    feat_max = feat.max(0).values
    feat_min = feat.min(0).values
    labels = torch.LongTensor(labels_np).to(device)
    degree = adj.sum(1)
    surro_net = GCN(features.shape[1], 64, labels.max().item() + 1, 0.5).float().to(device)
    surro_net.load_state_dict(torch.load(gcn_save_file+'_checkpoint.pt'))
    surro_net.eval()
    rep_net =  GIN(2, 2, features.shape[1], 64, labels.max().item()+1, 0.5, False, 'sum', 'sum').to(device)
    rep_net.load_state_dict(torch.load(rep_save_file+'_checkpoint.pt'))
    rep_net.eval()
    for p in surro_net.parameters():
        p.requires_grad = False
    print()
    
    split = np.aload('../../datasets/splits/' + args.dataset+ '_split.npy').item()
    index_train = split['train']
    index_val = split['val']
    index_test = split['test']
    
    ninj = index_test.shape[0]
    ratio = ninj/(n+ninj)
    new_n = n + ninj
    ground_truth = np.zeros(n)

    # Original graph:
    print('Original graph:')

    # atk_suc = mask_evaluate(adj, features, labels, surro_net, np.ones_like(n), index_test, device)
    # print('No mask: attack success rate:',atk_suc)
    print()
    # print()
    # graphpath = '../../GNIA_DINA/new_graphs/' + args.dataset + '/ablat/' 
    # graphpath = '../../TDGIA_DINA/new_graphs/' + args.dataset + '/ablat/' 
    graph_save_file = get_filelist(graphpath, [], name='')

    print('graph_save_file',graph_save_file)
    ground_truth = np.ones(n+ninj)
    ground_truth[n:] = 0
    detect_name_arr = ['copod', 'pca', 'hbos', 'iforest', 'ae', 'AVG']
    # detect_name_arr = ['copod', 'pca', 'hbos', 'iforest', 'AVG']

    for graph in graph_save_file:
        # graph_name = 'original_graph'
        # ground_truth = np.ones(n)
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        adj, features, labels_np = load_npz(graph)
        adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        nor_adj_tensor = normalize_tensor(adj_tensor)
        feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
        # emb, _ = rep_net(feat, nor_adj_tensor)
        # emb = emb.detach().cpu().numpy()
        emb = features.toarray()
        # print('emb',emb.shape, emb)
        # Probabilistic
        copod = COPOD(contamination=ratio)
        gmm = GMM(contamination=ratio)
        # Linear Model
        pca = PCA(n_components=2, contamination=ratio)
        # Proximity-Based
        cblof = CBLOF(contamination=ratio)
        hbos = HBOS(contamination=ratio)
        knn = KNN(contamination=ratio)
        # Outlier Ensembles
        iforest = IForest(contamination=ratio)
        # neural networks
        ae = AutoEncoder(contamination=ratio, verbose=0)
        i = 0
        acc_fake_arr = np.zeros(len(detect_name_arr))
        acc_all_arr = np.zeros(len(detect_name_arr))
        atk_suc_arr = np.zeros(len(detect_name_arr))
        # for model in [copod, pca, hbos, iforest]:
        for model in [copod, pca, hbos, iforest, ae]:
            detect_name = str(model).split('(')[0]
            print(detect_name)
            st = time.time()
            model.fit(emb)
            print('during', time.time()-st)
            out = model.labels_
            row_0 = np.where(out == 1)[0]
            row_1 = np.where(out == 0)[0]
            if row_0.shape[0] < row_1.shape[0]:
                out = 1 - out
            acc_fake = (ground_truth[n:]==out[n:]).sum()/len(out[n:])
            acc_all = (ground_truth==out).sum()/len(out)
            atk_suc = mask_evaluate(adj, features, labels, surro_net, out, index_test, device)
            print("Test misclassification {:.4%}".format(atk_suc))
            print("Test acc fake {:.4%}".format(acc_fake))
            print("Test acc all {:.4%}".format(acc_all))
            acc_fake_arr[i], acc_all_arr[i], atk_suc_arr[i] = acc_fake, acc_all, atk_suc
            print()
            i += 1
        acc_fake_arr[-1] = acc_fake_arr[:-1].mean()
        acc_all_arr[-1] = acc_all_arr[:-1].mean()
        atk_suc_arr[-1] = atk_suc_arr[:-1].mean()
        file = 'log/' + args.dataset + '/' +  args.suffix + '.csv'
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([graph_name])
        
        dataframe = pd.DataFrame({u'detect_name':detect_name_arr,u'acc_fake':acc_fake_arr,u'acc_all':acc_all_arr,u'atk_suc':atk_suc_arr})
        dataframe.to_csv(file, mode='a')
        print('*'*30)
    return


if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='GNIA')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--postprocess', type=bool, default=False, help='whether have sigmoid and rescale')

    # dataset
    parser.add_argument('--dataset', default='ogbarxiv',help='dataset to use')
    
    parser.add_argument('--surro_type', default='gcn',help='surrogate gnn model')
    parser.add_argument('--victim_type', default='gcn',help='evaluation gnn model')

    # optimization
    # parser.add_argument('--optimizer_G', choices=['Adam','SGD', 'RMSprop'], default='RMSprop',help='optimizer_G')
    parser.add_argument('--lr_G', default=1e-5, type=float, help='learning rate of Generator')
    parser.add_argument('--lr_D', default=1e-5, type=float, help='learning rate of Discriminator')
    parser.add_argument('--wd', default=0., type=float , help='weight decay')
    
    parser.add_argument('--attrtau', default=None, help='tau of gumbel softmax on attr')
    parser.add_argument('--edgetau', default=0.01, help='tau of gumbel softmax on edge')
    
    parser.add_argument('--epsst', default=50, type=int, help='epsilon start: coefficient of the gumbel sampling')
    parser.add_argument('--epsdec', default=1, type=float, help='epsilon decay: coefficient of the gumbel sampling')
    parser.add_argument('--patience', default=50, type=int, help='patience of early stopping')
    parser.add_argument('--atk_stand', default=0.6, type=float, help='attack success standard')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--multiedge', default=False, type=bool, help='budget of edges connected to injected node')
    parser.add_argument('--allnodes', default=False, type=bool, help='budget of edges connected to injected node')
    
    parser.add_argument('--counter', type=int, default=0, help='counter')
    parser.add_argument('--best_score', type=float, default=0., help='best score')
    parser.add_argument('--st_iteration', type=int, default=0, help='start iteration')
    parser.add_argument('--niterations', type=int, default=100000, help='number of iterations')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--Dopt', type=int, default=1, help='Discriminator optimize Dopt times, G optimize 1 time.')
    parser.add_argument('--D_sn', type=str, default='none', choices=['none', 'SN'] , help='whether Discriminator use spectral_norm')
    parser.add_argument('--D_type', type=str, default='gin', help='Discriminator type')
    parser.add_argument('--loss_type', type=str, default='gan', choices=['gan', 'ns', 'hinge', 'wasserstein'], help='Loss type.')
    parser.add_argument('--num', type=int, default=4, help='Vote number.')
    parser.add_argument('--atk', type=str, default='eye')
    
    # parser.add_argument('--local_rank', type=int, default=2, help='DDP local rank')
    
    args = parser.parse_args()
    att_sucess = main(args) 


'''
hao
nohup python -u eval_classifier.py --suffix gfd  --gpu 0 --dataset ogbproducts > log/ogbproducts/gfd.log 2>&1 &
nohup python -u eval_classifier.py --suffix tdgia  --gpu 3 --dataset reddit >> log/reddit/tdgia.log 2>&1 &
nohup python -u eval_classifier.py --suffix tdgia+hao_d  --gpu 2 --dataset ogbarxiv >> log/ogbarxiv/tdgia+hao_d.log 2>&1 &

nohup python -u eval_classifier.py --suffix gnia+hao_d  --gpu 5 --dataset ogbproducts >> log/ogbproducts/gnia+hao_d.log 2>&1 &
nohup python -u eval_classifier.py --suffix gnia+hao_d  --gpu 4 --dataset reddit >> log/reddit/gnia+hao_d.log 2>&1 &
nohup python -u eval_classifier.py --suffix gnia+hao_d10  --gpu 1 --dataset ogbarxiv >> log/ogbarxiv/gnia+hao_d10.log 2>&1 &
nohup python -u eval_classifier.py --suffix gnia+cana  --gpu 1 --dataset ogbarxiv >> log/ogbarxiv/gnia+cana.log 2>&1 &
nohup python -u eval_classifier.py --suffix gnia+cana_new  --gpu 5 --dataset ogbarxiv >> log/ogbarxiv/gnia+cana_final.log 2>&1 &

nohup python -u eval_classifier.py --suffix pgd+hao_d  --gpu 2 --dataset ogbproducts >> log/ogbproducts/pgd+hao_d.log 2>&1 &
nohup python -u eval_classifier.py --suffix pgd+hao_d  --gpu 6 --dataset reddit >> log/reddit/pgd+hao_d.log 2>&1 &
nohup python -u eval_classifier.py --suffix pgd+hao_d  --gpu 0 --dataset ogbarxiv >> log/ogbarxiv/pgd+hao_d.log 2>&1 &
nohup python -u eval_classifier.py --suffix pgd+cana  --gpu 2 --dataset ogbarxiv >> log/ogbarxiv/pgd+cana.log 2>&1 &
nohup python -u eval_classifier.py --suffix pgd+cana_new22  --gpu 6 --dataset ogbarxiv >> log/ogbarxiv/pgd+cana_new22.log 2>&1 &

'''

