import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


def sparse_dense_mul(a,b,c):
    i=a._indices()[1]
    j=a._indices()[0]
    v=a._values()
    newv=(b[i]+c[j]).squeeze()
    newv=torch.exp(F.leaky_relu(newv))
    
    new=torch.sparse.FloatTensor(a._indices(), newv, a.size())
    return new
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
        x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
class TAGraph(nn.Module):
    def __init__(self, in_features, out_features,k=2,activation=None,dropout=False,norm=False):
        super(TAGraph, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin=nn.Linear(in_features*(k+1),out_features).cuda()
        self.norm=norm
        self.norm_func=nn.BatchNorm1d(out_features,affine=False)
        self.activation=activation
        self.dropout=dropout
        self.k=k
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        
    def forward(self,x,adj,dropout=0):
        
        fstack=[x]
        
        for i in range(self.k):
            y=torch.spmm(adj,fstack[-1])
            fstack.append(y)
        x=torch.cat(fstack,dim=-1)
        x=self.lin(x)
        if self.norm:
            x=self.norm_func(x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x
    
class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features,activation=None,dropout=False):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ll=nn.Linear(in_features,out_features).cuda()
        
        self.activation=activation
        self.dropout=dropout
    def forward(self,x,adj,dropout=0):
        
        x=self.ll(x)
      #  x=torch.spmm(adj,x)
        if not(self.activation is None):
            x=self.activation(x)
        if self.dropout:
            x=F.dropout(x,dropout)
        return x


class MLP(nn.Module):
    def __init__(self,num_layers,num_features):
        super(MLP, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(MLPLayer(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        x=x
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
        x=F.softmax(x, dim=-1)
        return x
class GCN(nn.Module):
    def __init__(self,num_layers,num_features,activation=F.elu):
        super(GCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=activation,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0):
        
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
       # x=F.softmax(x, dim=-1)
        return x
       
class GCN_norm(nn.Module):
    def __init__(self,num_layers,num_features):
        super(GCN_norm,self).__init__()
        self.GCN=GCN(num_layers,num_features)
        #self.ln=nn.LayerNorm(100).cuda()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
        #print(num_layers)
        
        for i in range(num_layers):
            self.layers.append(nn.LayerNorm(num_features[i]).cuda())
            if i!=num_layers-1:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1],activation=F.elu,dropout=True).cuda())
            else:
                self.layers.append(GraphConvolution(num_features[i],num_features[i+1]).cuda())
        #print(self.layers)
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        x=torch.clamp(x,min,max)
        for i in range(len(self.layers)):
            if i%2==1:
                x=self.layers[i](x,adj,dropout=dropout)
            else:
                x=self.layers[i](x)
       # x=F.softmax(x, dim=-1)
        return x
    
class rep(nn.Module):
    def __init__(self,num_features):
        super(rep,self).__init__()
        mid=num_features
        #mid=int(np.sqrt(num_features))+1
        #self.ln=nn.LayerNorm(100).cuda()
        self.num_features=num_features
        self.lin1=nn.Linear(num_features,mid)
        self.lin2=nn.Linear(mid,num_features)
        self.ln=nn.LayerNorm(mid)
        #self.att=nn.Linear(num_features,1)
        self.activation1=F.relu
        self.activation2=F.sigmoid
        #gain = nn.init.calculate_gain('relu')
        #nn.init.xavier_normal_(self.lin.weight, gain=gain)
        #print(num_layers)
        
        #print(self.layers)
        
    def forward(self,x,adj):
        '''
        att=self.att(x)
        att=self.activation1(att)
        att=F.softmax(att,dim=0)
        #print(att.size())
        '''
        sumlines=torch.sparse.sum(adj,[0]).to_dense()
        allsum=torch.sum(sumlines)
        avg=allsum/x.size()[0]
        att=sumlines/allsum
        att=att.unsqueeze(1)
        #print(att.size())
        
        normx=x/sumlines.unsqueeze(1)
        avg=torch.mm(att.t(),normx)
        #avg=torch.mean(x,dim=0,keepdim=True)
        
        y=self.lin1(avg)
        y=self.activation1(y)
        y=self.ln(y)
        y=self.lin2(y)
        y=self.activation2(y)
        y=0.25+y*2
        
        #print(x.size(),avg.size())
        dimmin=normx-avg  #n*100
        dimmin=torch.sqrt(att)*dimmin
        rep=torch.mm(dimmin.t(),dimmin) # 100*100
        #ones=torch.ones(x.size()).cuda() #n*100
        #ones=torch.sum(ones,dim=0,keepdim=True)
        covariance=rep
        #conv=covariance.unsqueeze(0)
        q=torch.squeeze(y)
        qq=torch.norm(q)**2
        ls=covariance*q
        ls=ls.t()*q
        diag=torch.diag(ls)
        sumdiag=torch.sum(diag)
        sumnondiag=torch.sum(ls)-sumdiag
        loss=sumdiag-sumnondiag/self.num_features
        diagcov=torch.diag(covariance)
        sumdiagcov=torch.sum(diagcov)
        sumnondiagcov=torch.sum(covariance)-sumdiagcov
        lscov=sumdiagcov-sumnondiagcov/self.num_features
        k=loss/lscov
        k=k*self.num_features/qq
        if not(self.training):
            #print(ls)
            print(k)
        #print(y.shape)
        x=x*y
        
        #print((z-x).norm())
        return x,k

class nonelayer(nn.Module):
    def __init__(self):
        super(nonelayer,self).__init__()
    def forward(self,x):
        return x
class TAGCN(nn.Module):
    def __init__(self,num_layers,num_features,k):
        super(TAGCN, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
            #print(num_layers)
            
        for i in range(num_layers):
            if i!=num_layers-1:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k,activation=F.leaky_relu,dropout=True).cuda())
            else:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k).cuda())
        #self.reset_parameters()
            #print(self.layers)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        #x=torch.clamp(x,min,max)
        #x=torch.atan(x)*2/np.pi
        for i in range(len(self.layers)):
            x=self.layers[i](x,adj,dropout=dropout)
           
        return x
class TArep(nn.Module):
    def __init__(self,num_layers,num_features,k):
        super(TArep, self).__init__()
        self.num_layers=num_layers
        self.num_features=num_features
        self.layers=nn.ModuleList()
            #print(num_layers)
            
        for i in range(num_layers):
            self.layers.append(rep(num_features[i]).cuda())
            if i!=num_layers-1:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k,activation=F.leaky_relu,dropout=True).cuda())
            else:
                self.layers.append(TAGraph(num_features[i],num_features[i+1],k).cuda())
            #print(self.layers)
        
    def forward(self,x,adj,dropout=0,min=-1000,max=1000):
        #x=torch.clamp(x,min,max)
        #x=torch.atan(x)*2/np.pi
        kk=0
        for i in range(len(self.layers)):
            
            if i%2==0:
                #x=self.layers[i](x)
                x,k=self.layers[i](x,adj)
                kk=k+kk
            else:
            #print(i,self.layers[i].lin.weight.norm(),x.shape)
                x=self.layers[i](x,adj,dropout=dropout)
            
        return x,kk
def GCNadj(adj,pow=-0.5):
    adj2=sp.eye(adj.shape[0])+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()


class sglayer(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(sglayer,self).__init__()
        self.lin=nn.Linear(in_feat,out_feat)
    def forward(self,x,adj,k=4):
        for i in range(k):
            x=torch.spmm(adj,x)
        x=self.lin(x)
        return x
        
class sgcn(nn.Module):
    def __init__(self,input_dim,output_dim,with_rep=False):
        super(sgcn,self).__init__()
        self.no=nn.BatchNorm1d(input_dim)
        self.in_conv=nn.Linear(input_dim,140)
        self.out_conv=nn.Linear(100,output_dim)
        self.act=torch.tanh
        self.layers=nn.ModuleList()
        self.with_rep=with_rep
        if with_rep:
            self.rep=nn.ModuleList()
            self.rep.append(rep(140))
            self.rep.append(rep(120))
            self.rep.append(rep(100))
        self.layers.append(sglayer(140,120))
        self.layers.append(nn.LayerNorm(120))
        self.layers.append(sglayer(120,100))
        self.layers.append(nn.LayerNorm(100))
        
        
    def forward(self,x,adj,dropout=0):
        x=self.no(x)
        x=self.in_conv(x)
        x=self.act(x)
        x=F.dropout(x,dropout)
        for i in range(len(self.layers)):
            
            if i%2==0:
                if self.with_rep:
                    x=self.rep[int(i/2)](x,adj)
                x=self.layers[i](x,adj)
            else:
                x=self.layers[i](x)
                x=self.act(x)
        if self.with_rep:
            x=self.rep[-1](x,adj)
        x=F.dropout(x,dropout)
        x=self.out_conv(x)
        
        return x
def SAGEadj(adj,pow=-1):
    adj2=sp.eye(adj.shape[0])*(1)+adj
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
        if (adj2.data[i]<0):
            adj2.data[i]=0
    adj2.eliminate_zeros()
    adj2 = sp.coo_matrix(adj2)
    if pow==0:
        return adj2.tocoo()
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=d_mat_inv_sqrt @ adj2
    
    return adj2.tocoo()

class tsail_sur(nn.Module):
    def __init__(self):
        super(tsail_sur,self).__init__()
        self.layers=nn.ModuleList()
        num_layers=3
        num_features=[100,200,128,128]
        for i in range(num_layers):
            self.layers.append(GCwithself(num_features[i],num_features[i+1],activation=F.relu,dropout=True).cuda())
            
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 18)
        )
    def forward(self,x,adj,dropout=0,min=0,max=0):
        
        for layer in self.layers:
            x=layer(x,adj,dropout=dropout)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
        
def tsail_pre(x):
    x=torch.clamp(x,-0.4,0.4)
    x.data[x.abs().ge(0.39).sum(1)>20]=0
    return x

class normSage(nn.Module):
    def __init__(self,in_features,pool_features,out_features,activation=None,dropout=False):
        super(normSage,self).__init__()
        self.pool_fc=nn.Linear(in_features,pool_features)
        self.fc1=nn.Linear(pool_features,out_features)
        self.fc2=nn.Linear(pool_features,out_features)
        self.activation=activation
        self.dropout=dropout
        
    def forward(self,x,adj,dropout=0,mu=2.0):
        
        x=F.relu(self.pool_fc(x))
        #usb=torch.max(x).data.item()
        
        #print(usb,usa)
        x3=x**mu
        x2=torch.spmm(adj,x3)**(1/mu)
        
        # In original model this is actually max-pool, but **10/**0.1 result in graident explosion. However we can still achieve similar performance using 2-norm.
        x4=self.fc1(x)
        x2=self.fc2(x2)
        x4=x4+x2
        if self.activation is not None:
            x4=self.activation(x4)
        if self.dropout:
            x4=F.dropout(x4,dropout)
        
        return x4
        


def cccn_adj(adj,pow=-0.5):
    adj2=adj+0
    for i in range(len(adj2.data)):
        if (adj2.data[i]>0 and adj2.data[i]!=1):
            adj2.data[i]=1
    adj2 = sp.coo_matrix(adj2)
    
    rowsum = np.array(adj2.sum(1))
    d_inv_sqrt = np.power(rowsum, pow).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj2=adj2+sp.eye(adj.shape[0])
    adj2=d_mat_inv_sqrt @ adj2 @ d_mat_inv_sqrt
    
    return adj2.tocoo()

class cccn_sur(nn.Module):
    def __init__(self):
        super(cccn_sur,self).__init__()
        self.gcn=GCN(3,[100,128,128,18],activation=F.relu)
    def forward(self,x,adj,dropout=0):
        x=torch.clamp(x,-1.8,1.8)
        x=self.gcn(x,adj,dropout=dropout)
        return x

class gcn_lm(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gcn_lm,self).__init__()
        self.ln=nn.LayerNorm(in_feat)
        self.gcn=GCN(4,[in_feat,256,128,64,out_feat])
    def forward(self,x,adj,dropout=0):
        #x=torch.clamp(x,-1.8,1.8)
        x=self.ln(x)
        x=self.gcn(x,adj,dropout=dropout)
        return x
        
class gcn(nn.Module):
    def __init__(self,in_feat,out_feat):
        super(gcn,self).__init__()
        #self.ln=nn.LayerNorm(in_feat)
        self.gcn=GCN(4,[in_feat,256,128,64,out_feat])
    def forward(self,x,adj,dropout=0):
        x=self.gcn(x,adj,dropout=dropout)
        return x
def daftstone_pre(adj,features):
    w=np.abs(features)
    a=np.sum(w>0.5,axis=1)
    b=np.sum(w>0.3,axis=1)
    idx=(a>35)+(b>60)
    m1=np.max(w,axis=1)
    for j in range(2):
        idx1=np.argmax(w,axis=1)
        for i in range(idx1.shape[0]):
            w[i,idx1[i]]=0
    
    m2=np.max(w,axis=1)
    idx1=(m1-m2<=0.002)
    idx2=np.where(m1==0)[0]
    idx1[idx2]=False
    scale=1.5
    dispersion = np.load("max_dispersion.npy")
    idx3 = np.sum(np.abs(features) > dispersion * scale, axis=1) !=0
    idx = np.where(idx + idx1 + idx3)[0]
    flag=np.zeros((len(features)), dtype=np.int)
    if (len(idx) != 0):
        features[idx,] = 0
        flag[idx]=1
    adj=adj.tocoo()
    adj.data[flag[adj.row]==1]=0
    adj.data[flag[adj.col]==1]=0
    adj=GCNadj(adj)
    return adj
    