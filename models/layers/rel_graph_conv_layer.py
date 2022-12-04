#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 11:24
# @Author  : Allen Xiong
# @File    : rel_graph_conv_layer.py
import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn


class RelGraphConvLayer(nn.Module):
    def __init__(self,in_feat, out_feat, rel_names, num_bases, weight=True, bias=True, activation=None, self_loop=True, dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(self.in_feat, self.out_feat, norm='right', weight=False, bias=True)
            # rel: dglnn.EdgeConv(self.in_feat, self.out_feat, batch_norm=True)
            for rel in self.rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(rel_names) and weight

        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), self.in_feat, self.out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(self.out_feat))
            nn.init.zeros_(self.h_bias)

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        '''

        :param g: DGLHeteroGraph, input graph
        :param inputs: dict[str, tensor], node feature for each node type
        :return:
        dict[str, tensor], New node feature for each node type
        '''
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k : v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelGraphEmbed(nn.Module):
    '''
    Embedding layer for featureless heterograph
    '''
    def __init__(self, g, embed_size, embed_name='embed', activation=None, dropout=0.0):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.embeds = nn.ParameterDict()
        for ntype in self.g.ntypes:
            embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self):
        return self.embeds


