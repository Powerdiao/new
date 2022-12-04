#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/13 12:55
# @Author  : Allen Xiong
# @File    : han.py.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import GATConv
from .layers.han_layer import HANLayer

class HAN(nn.Module):
    def __init__(self, meta_paths, in_dim, hidden_dim, out_dim, ntypes, etypes, num_heads=[4], dropout=0.2):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_dim, hidden_dim, etypes, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_dim * num_heads[l - 1],
                                        hidden_dim, etypes, num_heads[l], dropout))

        self.out_layer = nn.ModuleDict()
        for k in ntypes:
            self.out_layer[k] = nn.Linear(hidden_dim * num_heads[-1], out_dim)

    def forward(self, g, h):
        h = h[0]
        for block, layer in zip(g, self.layers):
            h = layer(block, h)
            # for k in h.keys():
            #     print('-'*50)
            #     print(h[k].shape)
            #     print('-' * 50)
        for k in h.keys():
            h[k] = self.out_layer[k](h[k])
            # print(k, h[k].shape)
        return h

    def totrain(self, positive_graph, negative_graph, blocks, x):
        outs = self.forward(blocks[0], x)
        print(outs)
        pos_score = self.predict(positive_graph, outs)
        neg_score = self.predict(negative_graph, outs)
        return pos_score, neg_score

    def predict(self, g, embeds):
        # return self.pred(g, embeds)
        with g.local_scope():
            g.ndata['x'] = embeds
            scores = {}
            for etype in g.canonical_etypes:
                g.apply_edges(dgl.function.u_mul_v('x', 'x', 'score'), etype=etype)
                tmp = torch.mv(g.edata['score'][etype], self.rel_embedding[etype].squeeze())
                # g.edata['score'][etype] = tmp
                scores[etype] = tmp
            return scores