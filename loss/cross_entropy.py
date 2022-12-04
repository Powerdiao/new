#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:16
# @Author  : Allen Xiong
# @File    : cross_entropy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.cross_entropy(score, label.float())
        return loss