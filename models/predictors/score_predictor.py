#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:39
# @Author  : Allen Xiong
# @File    : score_predictor.py
import dgl
import torch.nn as nn


class ScorePredictor(nn.Module):
    def __init__(self, margin):
        super(ScorePredictor, self).__init__()
        self.margin = margin

    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            if '_N' in x:
                edge_subgraph.ndata['x'] = x['_N']#[edge_subgraph.ndata['_ID']]
            else:
                edge_subgraph.ndata['x'] = x

            edge_subgraph = self.apply_edges(edge_subgraph, x)
            return edge_subgraph.edata['score']

    def apply_edges(self, edge_subgraph, x):
        for etype in edge_subgraph.canonical_etypes:
            if etype[0] in x and etype[-1] in x:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            else:
                edge_subgraph.edata['score']['etype'] = -1.0 * self.margin
        return edge_subgraph
