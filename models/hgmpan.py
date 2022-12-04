#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 12:45
# @Author  : Allen Xiong
# @File    : hgmpan.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.hgt_layer import HGTLayer

class HGMPAN(nn.Module):
    def __init__(self, g, ntypes, etypes, in_dim, hidden_dim, out_dim, num_layers, num_heads, dropout=0.2, use_norm=True):
        super(HGMPAN, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_norm = use_norm

        self.adapt_ws   = nn.ModuleDict()
        self.gcs        = nn.ModuleList()

        self.node_dict = {}
        self.edge_dict = {}

        for ntype in self.ntypes:
            self.node_dict[ntype] = len(self.node_dict)

        for etype in self.etypes:
            self.edge_dict[etype] = len(self.edge_dict)

        for ntype in ntypes:
            self.adapt_ws[ntype] = nn.Linear(self.in_dim, self.hidden_dim)

        for i in range(self.num_layers):
            self.gcs.append(HGTLayer(self.hidden_dim, self.hidden_dim, self.node_dict, self.edge_dict, num_heads[i], dropout, use_norm))

        self.out = nn.ModuleDict()
        for ntype in self.ntypes:
            self.out[ntype] = nn.Linear(self.hidden_dim, self.out_dim)



    def forward(self, g, h):
        # h = h[0]
        for ntype in self.ntypes:
            h[0][ntype] = F.leaky_relu(self.adapt_ws[ntype](h[0][ntype]))
            h[1][ntype] = F.leaky_relu(self.adapt_ws[ntype](h[1][ntype]))
        for i in range(self.num_layers):
            h = self.gcs[i](g[i], h)
            if i != self.num_layers - 1:
                t = g[i+1].dstdata['feat']
                for ntype in t.keys():
                    t[ntype] = F.leaky_relu(self.adapt_ws[ntype](t[ntype]))
                h = (h, t)

        for ntype in self.ntypes:
            h[ntype] = self.out[ntype](h[ntype])
        return h