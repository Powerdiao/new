#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 20:09
# @Author  : Allen Xiong
# @File    : hinsage.py.py
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn


class HinSAGE(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_hidden_layers, dropout=0.0, device=None):
        super(HinSAGE, self).__init__()

        self.g = g
        self.rel_names = self.g.etypes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList()

        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(self.in_dim, self.in_dim, aggregator_type='pool', feat_drop=dropout)
            for rel in self.rel_names
        }))

        for i in range(self.num_hidden_layers):
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.SAGEConv(self.in_dim, self.in_dim, aggregator_type='pool', feat_drop=dropout)
                for rel in self.rel_names
            }))

        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(self.in_dim, self.out_dim, aggregator_type='pool', feat_drop=dropout)
            for rel in self.rel_names
        }))

    def forward(self, g, h):
        # h = h[0]

        for layer, block in zip(self.layers, g):
            h = layer(block, h)
        return h


