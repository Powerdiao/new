# @Time     : 2021/5/2 23:12
# @Author   : Allen Xiong
# @File     : hetgnn.py
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn

class HetGNN(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_hidden_layers, dropout=0.0, device=None):
        super(HetGNN, self).__init__()

        self.g = g
        self.rel_names = self.g.etypes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.conv.GraphConv(self.in_dim, self.in_dim, norm='both')
            for rel in self.rel_names
        }))
        for i in range(self.num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.conv.GraphConv(self.in_dim, self.in_dim, norm='both')
                for rel in self.rel_names
            }))

        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.conv.GraphConv(self.in_dim, self.out_dim, norm='both')
            for rel in self.rel_names
        }))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        h = h[0]
        for layer, block in zip(self.layers, g):
            h = layer(block, h)
            for k in h.keys():
                h[k] = self.dropout(h[k])

        return h