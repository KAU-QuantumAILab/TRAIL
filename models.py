import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SGConv, GCNConv
import torch

class AE_GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, num_layer, alpha, model, leaky_slop, momentum):
        super(AE_GNN, self).__init__()

        self.num_layers = num_layer             # num of layers
        self.dropout = dropout                  # dropout
        self.alpha = alpha                      # hyperparameter, rate of average
        self.leaky_slop = leaky_slop            # leaky relu negative slop
        self.momentum = momentum                # batch normalization momentum
        self.model = model                      # graph model name
        self.num_classes = nclass               # num of class for output layer dimension
        self.gnn_layer = nn.ModuleList()        # gnn layers
        self.b_norm_list = nn.ModuleList()      # batch norm layer
        self.avg_lin = nn.Linear(nhid, nhid)    # for learn average vector
        self.layer_norm = nn.LayerNorm(nhid)    # Norm
        
        # Pile the layer
        
        # method on GCN
        if self.model == 'GCN':
            if self.num_layers == 1:    # only one layer
                self.last_layer= GCNConv(nfeat, nclass)
                self.b_norm = nn.BatchNorm1d(nclass, momentum= self.momentum)
                
            else:   # two layers
                self.gnn_layer.append(GCNConv(nfeat, nhid))
                self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))
                
                if self.num_layers > 2:  # more than two
                    for _ in range(self.num_layers-2):
                        self.gnn_layer.append(GCNConv(nhid, nhid))
                        self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))   
                                                    
                self.last_layer = GCNConv(nhid, nclass)

        # method on GraphSage
        elif self.model == 'GraphSage':

            if self.num_layers == 1:
                self.last_layer= SAGEConv(nfeat, nclass)
                self.b_norm = nn.BatchNorm1d(nclass, momentum=0.05)
                
            else:
                self.gnn_layer.append(SAGEConv(nfeat, nhid))
                self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))
                
                if self.num_layers > 3:
                    for _ in range(self.num_layers-2):
                        self.gnn_layer.append(SAGEConv(nhid, nhid))
                        self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))   
                                     
                self.last_layer = SAGEConv(nhid, nclass)

        # method on SGC
        elif self.model == 'SGC':

            if self.num_layers == 1:
                self.last_layer= SGConv(nfeat, nclass)
                self.b_norm = nn.BatchNorm1d(nclass, momentum=0.05)
                
            else:
                self.gnn_layer.append(SGConv(nfeat, nhid))
                self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))
                
                if self.num_layers > 3:
                    for _ in range(self.num_layers-2):
                        self.gnn_layer.append(SGConv(nhid, nhid))
                        self.b_norm_list.append(nn.BatchNorm1d(nhid, momentum = self.momentum))   

                self.last_layer = SGConv(nhid, nclass)
            

    def forward(self, x, adj):
        pile_emb = []
        
        # We don't use the average embedding on one layer. Model begin to use AE from 2 layer.
        if self.num_layers == 1:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.last_layer(x, adj)
            x = self.b_norm(x)
            x = F.leaky_relu(x, self.leaky_slop)
        else:
            # GNN Block
            for i in range(self.num_layers-1):
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gnn_layer[i](x, adj)
                x = self.b_norm_list[i](x)
                x = F.leaky_relu(x, self.leaky_slop)
                pile_emb.append(x)

            avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
            avg_vec = self.layer_norm(self.avg_lin(avg_vec))

            x = x + self.alpha*(avg_vec - x)
            x = self.last_layer(x, adj)

        return F.log_softmax(x, dim=1)

# ablation study -------------------------------------------------------------
# if you want to check the performance of ablation study, replace the below block code into foward function.
''' vanilla model -> use layer normalization instead of batch normalization
self.l_norm_list = nn.ModuleList()
self.l_norm_list.append(nn.LayerNorm(nhid))

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.layer_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.l_norm_list[i](x) 
            x = F.relu(x)

        x = self.last_layer(x, adj)
        return F.log_softmax(x, dim=1)

'''
# B.L.-------------------------------------------------------
''' Batch Norm + leaky relu

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.leaky_relu(x, self.leaky_slop)
    else:
        for i in range(self.num_layers-1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.leaky_relu(x, self.leaky_slop)

        x = self.last_layer(x, adj)

    return F.log_softmax(x, dim=1)

'''
# B.L.A -----------------------------------------------------
''' + average

pile_emb = []

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.leaky_relu(x, self.leaky_slop)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.leaky_relu(x, self.leaky_slop)
            pile_emb.append(x)

        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        x = avg_vec

        x = self.last_layer(x, adj)

    return F.log_softmax(x, dim=1)

'''
# N.N. -----------------------------------------------------
''' not use norm

pile_emb = []
    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.leaky_relu(x, self.leaky_slop)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.leaky_relu(x, self.leaky_slop)
            pile_emb.append(x)

        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        avg_vec = self.avg_lin(avg_vec)

        x = x - self.alpha*(x - avg_vec)
        x = self.last_layer(x, adj)

    return F.log_softmax(x, dim=1)

'''
# L-R ------------------------------------------------------
''' leaky relu -> relu

pile_emb = []
    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.relu(x)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.relu(x)
            pile_emb.append(x)
            
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        avg_vec = self.layer_norm(self.avg_lin(avg_vec))

        x = x - self.alpha*(x - avg_vec)
        x = self.last_layer(x, adj)

    return F.log_softmax(x, dim=1)

'''
# AE(our method) -------------------------------------------
'''
pile_emb = []

    if self.num_layers == 1:
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.last_layer(x, adj)
        x = self.b_norm(x)
        x = F.leaky_relu(x, self.leaky_slop)
    else:
        for i in range(self.num_layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gnn_layer[i](x, adj)
            x = self.b_norm_list[i](x)
            x = F.leaky_relu(x, self.leaky_slop)
            pile_emb.append(x)
    
        avg_vec = torch.mean(torch.stack(pile_emb, dim=0), dim=0)
        avg_vec = self.layer_norm(self.avg_lin(avg_vec))

        x = x + self.alpha*(avg_vec - x)
        x = self.last_layer(x, adj)

    return F.log_softmax(x, dim=1)
    
'''