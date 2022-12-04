#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 0:05
# @Author  : Allen Xiong
# @File    : hetero_dotproduct_predictor.py
import dgl.function as fn
import torch.nn as nn

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']