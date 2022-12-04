#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:41
# @Author  : Allen Xiong
# @File    : rgcn.py
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictors import ScorePredictor, DisMult
from .layers.stochastic_twoLayer_rgcn import StochasticTwoLayerRGCN
from .layers.rel_graph_conv_layer import RelGraphConvLayer
from .layers import HeterRGCNLayer

from tqdm import tqdm

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_hidden_layers, canonical_etypes, device=None):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.canonical_etypes = canonical_etypes

        self.layers = nn.ModuleList()

        self.layers.append(HeterRGCNLayer(self.in_dim, self.hidden_dim, self.canonical_etypes))

        for i in range(self.num_hidden_layers):
            self.layers.append(HeterRGCNLayer(self.hidden_dim, self.hidden_dim, self.canonical_etypes))

        self.layers.append(HeterRGCNLayer(self.hidden_dim, self.out_dim, self.canonical_etypes))

    def forward(self, g, h):
        '''
        :param blocks: list, contains nodes and edges from sampler's layers, each element is the layer sampled subgraph
        :param x: node features
        :return x: representation of input nodes
        '''
        h = h[0]
        for layer, block in zip(self.layers, g):
            h = layer(block, h)
            h = {k: nn.functional.leaky_relu(v) for k, v in h.items()}
        return h

    def totrain(self, positive_graph, negative_graph, blocks, x):
        '''
        :param positive_graph: graph sampled from real graph
        :param negative_graph: the edges never contain in real graph
        :param blocks: list, contains nodes and edges from sampler's layers, each element is the layer sampled subgraph
        :param x: node features
        :return pos_score: the positive_graph's score
        :return neg_score: the negative_graph's score
        '''
        pass
        # x = self.rgcn(blocks, x)
        # pos_score = self.pred(positive_graph, x)
        # neg_score = self.pred(negative_graph, x)
        # return pos_score, neg_score


class RGCN_Hetero_Entity_Classify(nn.Module):
    def __init__(self, g, hidden_dim, out_dim, num_bases, num_hidden_layers=1, dropout=0.0, use_self_loop=False, device=None):
        super(RGCN_Hetero_Entity_Classify, self).__init__()
        self.g = g
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers

        self.rel_names = list(set(self.g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases

        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # input to hidden
        self.layers.append(RelGraphConvLayer(
            in_feat=self.hidden_dim,
            out_feat=self.hidden_dim,
            rel_names=self.rel_names,
            num_bases=self.num_bases,
            activation=F.relu,
            self_loop=self.use_self_loop,
            dropout=self.dropout
            ))

        # hidden layers
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                in_feat=self.hidden_dim,
                out_feat=self.hidden_dim,
                rel_names=self.rel_names,
                num_bases=self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout
            ))

        # hidden to output
        self.layers.append(RelGraphConvLayer(
            in_feat=self.hidden_dim,
            out_feat=self.out_dim,
            rel_names=self.rel_names,
            num_bases=self.num_bases,
            activation=F.relu,
            self_loop=self.use_self_loop,
            dropout=self.dropout
        ))

    def forward(self, g, inputs):
        '''
        :param g: sample blocks of graph
        :param inputs:
        :return:
        '''
        h = inputs[0]
        for layer, block in zip(self.layers, g):
            h = layer(block, h)
        return h

    def inference(self, g, batch_size, device, x):
        for l, layer in enumerate(self.layers):
            y = {ntype: torch.zeros(
                    g.number_of_nodes(ntype),
                    self.hidden_dim if l != len(self.layers) - 1 else self.out_dim)
                for ntype in g.ntypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, {k: torch.arange(g.number_of_nodes(k)) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                device=device
            )

            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                blocks = [block.to(device) for block in blocks]
                h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}

                h = layer(blocks[0], h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y


class RGCN_Hetero_Link_Prediction(RGCN_Hetero_Entity_Classify):
    def __init__(self, g, hidden_dim, out_dim, num_bases, num_hidden_layers=1, dropout=0.0, use_self_loop=False, device=None):
        super(RGCN_Hetero_Link_Prediction, self).__init__(g, hidden_dim, out_dim, num_bases, num_hidden_layers, dropout, use_self_loop)
