'''
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            # self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                # h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)




class GIN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GIN, self).__init__()

        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        # self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            # self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.new_linear_prediction = torch.nn.ModuleList()
        self.new_linear_prediction = MLP(num_mlp_layers, input_dim+(num_layers)*hidden_dim, hidden_dim, output_dim)
        # for layer in range(num_layers):
        #     if layer == 0:
        #         self.linears_prediction.append(nn.Linear(input_dim, output_dim))
        #     else:
        #         self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    def next_layer(self, h, layer, g = None, value=None):
        ###pooling neighboring nodes and center nodes altogether  

        pooled = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=h, rhs_data=value)
        # pooled = torch.spmm(Adj_block, h)

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)
        # h = self.batch_norms[layer](pooled_rep)
        h = pooled_rep
        #non-linearity
        h = F.relu(h)
        return h


    def forward(self, feat, adj_tensor):
        #list of hidden representation at each layer (including input)
        hidden_rep = [feat]
        h = feat
        try:
            row, column = adj_tensor.coalesce().indices()
            g = dgl.graph((column, row), num_nodes=adj_tensor.shape[0], device=adj_tensor.device)
            value = adj_tensor.coalesce().values()
        except:
            row, column, value = adj_tensor.coo()
            g = dgl.graph((column, row), num_nodes=adj_tensor.size(0), device=adj_tensor.device())

        
        for layer in range(self.num_layers):
            if layer == 0:
                h = self.mlps[layer](h)
            else:
                h = self.next_layer(h, layer,g = g,value=value) 
            # h = self.next_layer(h, layer, Adj_block = adj_tensor, g = g)
            hidden_rep.append(h)

        #perform pooling over all nodes in each graph in every layer
        hidden_rep = F.dropout(torch.cat(hidden_rep, 1), self.final_dropout, training=self.training)
        output = self.new_linear_prediction(hidden_rep)
        # output = 0
        # for layer, h in enumerate(hidden_rep):
        #     # pooled_h = torch.spmm(graph_pool, h)
        #     # 先不对节点求和
        #     output += F.dropout(self.linears_prediction[layer](h), self.final_dropout, training = self.training)

        return hidden_rep, output
