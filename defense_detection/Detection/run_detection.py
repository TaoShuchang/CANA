import sys
import os
import csv
import torch
import argparse
import numpy as np
import scipy.sparse as sp

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

import os,sys

os.chdir(sys.path[0])
sys.path.append('../..')
from utils import *
from modules.eval_metric import *
from gnn_model.gcn import GCN
from gnn_model.gin import GIN


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
    csv_file = 'logs/' + args.dataset + '_' +  args.suffix + '.csv'
    graphpath = '../../' + args.suffix + '_graphs/' + args.dataset + '/' 

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('dataset',args.dataset)
    adj, features, labels_np = load_npz(f'../../datasets/{args.dataset}.npz')
    n = adj.shape[0]
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n)
    adj[adj > 1] = 1
    lcc = largest_connected_components(adj)
    adj = adj[lcc][:, lcc]
    features = features[lcc]
    labels_np = labels_np[lcc]
    n = adj.shape[0]
    print('Nodes num:',n)

    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)

    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
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
    print()
    # print()
    graph_save_file = get_filelist(graphpath, [], name='')

    print('graph_save_file',graph_save_file)
    ground_truth = np.ones(n+ninj)
    ground_truth[n:] = 0
    detect_name_arr = ['copod', 'pca', 'hbos', 'iforest', 'ae']

    for graph in graph_save_file:
        # graph_name = 'original_graph'
        # ground_truth = np.ones(n)
        graph_name = graph.split('/')[-1]
        print('inject attack',graph_name)
        adj, features, labels_np = load_npz(graph)
        emb = features.toarray()
        # Probabilistic
        copod = COPOD(contamination=ratio)
        # Linear Model
        pca = PCA(n_components=2, contamination=ratio)
        # Proximity-Based
        hbos = HBOS(contamination=ratio)
        # Outlier Ensembles
        iforest = IForest(contamination=ratio)
        # Neural Networks
        ae = AutoEncoder(contamination=ratio, verbose=0)
        i = 0
        acc_fake_arr = np.zeros(len(detect_name_arr))
        acc_all_arr = np.zeros(len(detect_name_arr))
        atk_suc_arr = np.zeros(len(detect_name_arr))
        for model in [copod, pca, hbos, iforest, ae]:
            detect_name = str(model).split('(')[0]
            print(detect_name)
            model.fit(emb)
            out = model.labels_
            row_0 = np.where(out == 1)[0]
            row_1 = np.where(out == 0)[0]
            if row_0.shape[0] < row_1.shape[0]:
                out = 1 - out
            atk_suc = mask_evaluate(adj, features, labels, surro_net, out, index_test, device)
            print("Test misclassification {:.4%}".format(atk_suc))
            atk_suc_arr[i] = atk_suc
            print()
            i += 1

        # atk_suc_arr[-1] = atk_suc_arr[:-1].mean()
        with open(csv_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([graph_name])
        
        dataframe = pd.DataFrame({u'detect_name':detect_name_arr,u'acc_fake':acc_fake_arr,u'acc_all':acc_all_arr,u'atk_suc':atk_suc_arr})
        dataframe.to_csv(csv_file, mode='a')
        print('*'*30)
    return


if __name__ == '__main__':
    setup_seed(123)
    parser = argparse.ArgumentParser(description='Detect')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--gpu', type=str, default="1", help='GPU ID')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--postprocess', type=bool, default=False, help='whether have sigmoid and rescale')

    # dataset
    parser.add_argument('--dataset', default='ogbarxiv',help='dataset to use')
       
    args = parser.parse_args()
    att_sucess = main(args) 


