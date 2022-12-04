#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/18 13:37
# @Author  : Allen Xiong
# @File    : DisMult.py
import dgl
import torch
import torch.nn as nn


class DisMult(nn.Module):
    def __init__(self, canonical_etypes, dim, device):
        super(DisMult, self).__init__()
        self.canonical_etypes = canonical_etypes
        self.num_rels = len(self.canonical_etypes)
        self.dim = dim
        self.device = device
        self.rel_embedding = {}
        self.rel_embedding['_no_relation'] = nn.Parameter(torch.empty(self.dim, requires_grad=True).to(device))
        for k in self.canonical_etypes:
            self.rel_embedding[k] = nn.Parameter(torch.empty(self.dim, requires_grad=True).to(device))

        for k in self.rel_embedding.keys():
            nn.init.xavier_uniform_(self.rel_embedding[k], gain=nn.init.calculate_gain('relu'))

    def forward(self, g, node_embeds):
        with g.local_scope():
            g.ndata['x'] = node_embeds
            scores = {}
            for etype in g.canonical_etypes:
                # g.apply_edges(self._calc, etype=etype, inplace=False)
                g.apply_edges(dgl.function.u_mul_v('x', 'x', 'score'), etype=etype)
                # calculate score for each edge
                tmp = torch.mv(g.edata['score'][etype], self.rel_embedding[etype].squeeze())
                # calculate scores for each type of edge, the src and dst keep no change
                # tmp = torch.matmul(g.edata['score'][etype], rels_embedding.T)
                scores[etype] = tmp
            return scores

    def _calc(self, edges):
        if edges.src['x'].shape[0] == 0:
            dic = {'score': torch.Tensor(0).to(self.device)}
        else:
            dic = {'score': torch.cat([
                torch.matmul(
                    edges.src['x'][i] * self.rel_embedding[edges.canonical_etype[1]],
                    edges.dst['x'][i])
                for i in range(edges.src['x'].shape[0])])
            }
        return dic





