'''
Refer to https://github.com/THUDM/tdgia
'''

import os
import sys
import copy
import time
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.nn as nn
from gcn import *
from uesd_models import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, LambdaLR

import sys
sys.path.append('../')
from utils import *
from modules.eval_metric import *
from modules.losses import compute_gan_loss,percept_loss, compute_D_loss

def generateaddon(weight1, weight2, applabels, culabels, adj, origin, cur, testindex, addmax=500, num_classes=18, connect=65, sconnect=20):
    # applabels: 50000
    # culabels: confidency of 50000*18
    newedgesx=[]
    newedgesy=[]
    newdata=[]
    thisadd=0
    num_test=len(testindex)
    import random
    # culabel 是过了softmax的output 维度 n*nc, nc是label类别数
    addscore=np.zeros(num_test)
    deg=np.array(adj.sum(axis=0))[0]+1.0
    normadj=GCNadj(adj)
    normdeg=np.array(normadj.sum(axis=0))[0]
    # 对每个目标节点计算
    for i in range(len(testindex)):
        it=testindex[i]
        label=applabels[it]             
        score=culabels[it][label]+2     
        addscore1=score/deg[it]        
        addscore2=score/np.sqrt(deg[it]) 
        sc=weight1*addscore1+weight2*addscore2/np.sqrt(connect+sconnect) 
        addscore[i]=sc

    # higher score is better
    sortedrank=addscore.argsort()
    sortedrank=sortedrank[-addmax*connect:]
    
    labelgroup=np.zeros(num_classes)
    #separate them by applabels
    labelil=[]
    for i in range(num_classes):
        labelil.append([])
    random.shuffle(sortedrank)
    for i in sortedrank:
        label=applabels[testindex[i]]    
        labelgroup[label]+=1             
        labelil[label].append(i)        

    pos=np.zeros(num_classes)
    for i in range(addmax):
        for j in range(connect):
            smallest=1
            smallid=0
            for k in range(num_classes):
                if len(labelil[k])>0:
                    if (pos[k]/len(labelil[k]))<smallest :
                        smallest=pos[k]/len(labelil[k])
                        smallid=k
            tu=labelil[smallid][int(pos[smallid])]
            
            #return to random
            #tu=sortedrank[i*connect+j]
            pos[smallid]+=1
            x=cur+i
            y=testindex[tu]
            newedgesx.extend([x,y])
            newedgesy.extend([y,x])
            newdata.extend([1,1])
                
    islinked=np.zeros((addmax,addmax))
    for i in range(addmax):
        j=np.sum(islinked[i])
        rndtimes=100
        while (np.sum(islinked[i])<sconnect and rndtimes>0):
            x=i+cur
            rndtimes=100
            yy=random.randint(0,addmax-1)
                
            while (np.sum(islinked[yy])>=sconnect or yy==i or islinked[i][yy]==1) and (rndtimes>0):
                yy=random.randint(0,addmax-1)
                rndtimes-=1
                    
            if rndtimes>0:
                y=cur+yy
                islinked[i][yy]=1
                islinked[yy][i]=1
                newedgesx.extend([x,y])
                newedgesy.extend([y,x])
                newdata.extend([1,1])
                
    thisadd=addmax
            
    print('in thisadd',thisadd, 'len(newedgesx)', len(newedgesx))
    add1=sp.csr_matrix((thisadd,cur))
    add2=sp.csr_matrix((cur+thisadd,thisadd))
    adj=sp.vstack([adj,add1])
    adj=sp.hstack([adj,add2])
    adj.row=np.hstack([adj.row,newedgesx])
    adj.col=np.hstack([adj.col,newedgesy])
    adj.data=np.hstack([adj.data,newdata])
    return thisadd, adj


def getprocessedadj(adj,modeltype,feature=None):
    processed_adj=GCNadj(adj)
    if modeltype in ["graphsage_norm"]:
        processed_adj=SAGEadj(adj,pow=0)
    if modeltype=="graphsage_max":
        from dgl import DGLGraph
        from dgl.transform import add_self_loop
        dim2_adj=DGLGraph(adj)
        processed_adj=add_self_loop(dim2_adj).to('cuda')
    if modeltype=="rgcn":
        processed_adj=(processed_adj,GCNadj(adj,pow=-1))
    return processed_adj
    
def getprocessedfeat(feature,modeltype):
    feat=feature+0.0
    return feat
    
    
    
def getresult(adj_tensor, feat, model):

    model.eval()
    with torch.no_grad():
        result=model(feat, adj_tensor)
    return result

def checkresult(curlabels,testlabels,origin,testindex):
    evallabels=curlabels[testindex]
    tlabels=torch.LongTensor(testlabels).to(device)
    acc=(evallabels==tlabels)
    acc=acc.sum()/(len(testindex)+0.0)
    acc=acc.item()
    return acc
    
def buildtensor(adj):
    sparserow=torch.LongTensor(adj.row).unsqueeze(1)
    sparsecol=torch.LongTensor(adj.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow,sparsecol),1).to(device)
    sparsedata=torch.FloatTensor(adj.data).to(device)
    import copy
    adjtensor=torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(adj.shape)).to(device)
    return adjtensor
    
def getmultiresult(adj,features,models,modeltype,origin,testindex):
    
    pred=[]
    predb=[]
    iw=0
    with torch.no_grad():
        for i in range(len(models)):
            processed_adj=getprocessedadj(adj,modeltype[i],feature=features.data.cpu().numpy())
            if not(modeltype[i] in ['graphsage_max','rgcn']):
                adjtensor=buildtensor(processed_adj)
            if modeltype[i] =='graphsage_max':
                adjtensor=processed_adj
            if modeltype[i]=='rgcn':
                adjtensor=(buildtensor(processed_adj[0]),buildtensor(processed_adj[1]))
            feat=getprocessedfeat(features,modeltype[i])
            models[i].eval()
            # iza=models[i](feat,adjtensor,dropout=0)
            iza=models[i](feat,adjtensor)
           # izb=iza.argmax(-1).cpu().numpy()
            iza=F.softmax(iza,dim=-1)
            iw=iza+iw
            #pred.append(izb)
            #print(izb)
    
    
    surlabel=iw.argmax(-1).cpu().numpy()
    return surlabel


    
def trainaddon(thisadd, lr, epochs, new_adj_tensor, surro_net, cur_feat, adj_tensor, n, testindex,
                reallabels, sec, netD, rep_net, optimizer_D, Dopt, atk_alpha=1, alpha=10, beta=0.01, writer=None, in_epoch=0, Dopt_in_epoch=0, loss_type='gan'):
    device = cur_feat.device
    _, num_features = cur_feat.shape
    feat = cur_feat[:n]
    injected_feat = cur_feat[n:].cpu().data.numpy()
    injected_feat = torch.FloatTensor(injected_feat).to(device)
    #revert it back
    cur_inj_feat = torch.randn((thisadd, num_features)).to(device)
    # optimizer = torch.optim.Adam([{'params':[cur_inj_feat,injected_feat]}], lr=lr)
    # optimizer.zero_grad()
    if (thisadd+1)*50+1<epochs:
        epochs = (thisadd+1)*50+1
    ori_label = reallabels[testindex]
    best_wrong_label = sec[testindex]
    inj_feat = torch.cat((injected_feat, cur_inj_feat),0)
    new_n = n + inj_feat.shape[0]
    Dopt_out_epoch = Dopt_in_epoch
    bs = 800
    for epoch in range(epochs):
        out_epoch = in_epoch + epoch
        inj_feat.requires_grad_(True)
        optimizer=torch.optim.Adam([{'params':[inj_feat]}], lr=lr)
        surro_net.eval()
        feature_orc = feat + 0.0
        
        new_feat = torch.cat((feature_orc, inj_feat),0)
        pred = surro_net(new_feat, normalize_tensor(new_adj_tensor))
        netD.train()
        for Dopt_ep in range(Dopt):
            if new_n - n > bs:
                fake_batch =  np.random.randint(n, new_n, (bs))
            else:
                fake_batch = np.arange(n,new_n)
            real_batch =  np.random.randint(0, adj_tensor.shape[0], (len(fake_batch)))
            train_loss_D, train_acc_D, train_acc_real, train_acc_fake = compute_D_loss(real_batch, fake_batch, netD, adj_tensor, feat, new_feat, new_adj_tensor)
            optimizer_D.zero_grad()
            train_loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.1)
            optimizer_D.step()
            train_loss_D_detach = train_loss_D.detach().item()
            del train_loss_D
            # print('Acc D', train_acc_D)
            # print('Loss D', train_loss_D_detach)
            writer.add_scalar('Each opt D/acc real D', train_acc_real, Dopt_out_epoch)
            writer.add_scalar('Each opt D/acc fake D', train_acc_fake, Dopt_out_epoch)
            writer.add_scalar('Each opt D/acc D', train_acc_D, Dopt_out_epoch)
            writer.add_scalar('Each opt D/loss D', train_loss_D_detach, Dopt_out_epoch)
            Dopt_out_epoch += 1
        
        netD.eval()
        # stablize the pred_loss, only if disguise_coe > 0
        atk_loss = F.relu(pred[testindex, ori_label] - pred[testindex, best_wrong_label])
        atk_loss = (atk_loss).mean()
        
        all_emb_list = rep_net(new_feat, new_adj_tensor)
        pred_fake_G = netD(new_feat, new_adj_tensor)[1]
        loss_G_fake = compute_gan_loss(loss_type, pred_fake_G[fake_batch])
        loss_G_div = percept_loss(all_emb_list, fake_batch)
        all_loss = atk_alpha * atk_loss + alpha * loss_G_fake + beta * loss_G_div 
        
        fake_label = torch.full((fake_batch.shape[0], 1), 0.0, device=pred_fake_G.device)
        acc_fake = netD.compute_acc(pred_fake_G[fake_batch], fake_label) 
        metric_atk_suc = ((reallabels[testindex] != pred[testindex].argmax(1)).sum()).item()/len(testindex)
        val_GFD = calculate_graphfd(rep_net, adj_tensor, feat, new_adj_tensor, new_feat, np.arange(n), np.arange(n, new_feat.shape[0]), verbose=False)
        
        writer.add_scalar('Metric/Atk success', metric_atk_suc, out_epoch)
        writer.add_scalar('Metric/GraphFD', val_GFD, out_epoch)
        writer.add_scalar('loss_Feat/Dacc', acc_fake, out_epoch)
        writer.add_scalar('loss_Feat/GAN', loss_G_fake, out_epoch)
        writer.add_scalar('loss_Feat/Div', loss_G_div, out_epoch)
        writer.add_scalar('loss_Feat/Attack', atk_loss, out_epoch)
        writer.add_scalar('loss_Feat/All', all_loss, out_epoch)
        
        if epoch%75==0:
            print("epoch:", epoch, "loss:", all_loss.item(), " misclassification:", metric_atk_suc)

        best_inj_feat = new_feat[n:].data  
        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        inj_feat = inj_feat.detach()


    return best_inj_feat, metric_atk_suc, val_GFD, out_epoch, Dopt_out_epoch
    