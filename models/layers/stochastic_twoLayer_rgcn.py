#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:43
# @Author  : Allen Xiong
# @File    : stochastic_twoLayer_rgcn.py
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn

class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                for rel in rel_names
            })
        self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                for rel in rel_names
            })

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x