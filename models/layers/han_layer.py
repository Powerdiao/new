#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 19:37
# @Author  : Allen Xiong
# @File    : han_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from collections import defaultdict
from dgl.nn.pytorch import GATConv

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0], ) + beta.shape)

        return (beta * z).sum(1)

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, etypes, layer_num_heads, dropout=0.2):
        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(dglnn.HeteroGraphConv({
                    etype: GATConv(in_size, out_size, layer_num_heads,
                        feat_drop=dropout, attn_drop=dropout, activation=None,
                        residual=False,allow_zero_in_degree=True)
                    for etype in etypes
                })
            )
            # self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
            #                                dropout, dropout, activation=F.elu,
            #                                allow_zero_in_degree=True))

        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = defaultdict(list)

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()

            for meta_path in self.meta_paths:
                # print(dgl.metapath_reachable_graph(g, meta_path))
                # self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)

                tmp_g = dgl.edge_type_subgraph(g, meta_path)
                self._cached_coalesced_graph[meta_path] = tmp_g

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            # print(meta_path)
            # if(not isinstance(h, tuple)):
            #     print(h["author"].shape)
            #     print(h['paper'].shape)
            # # print(new_g.dsttypes)
            # print(g.nodes[new_g.srctypes[0]].data['_ID'].shape)
            # print(new_g.nodes[new_g.srctypes[0]].data['_ID'].shape)
            #
            # print(g.nodes[new_g.dsttypes[0]].data['_ID'].shape)
            # print(new_g.nodes[new_g.dsttypes[0]].data['_ID'].shape)
            # print(new_g.srcdata['_ID'].shape)
            # print(new_g.dstdata['_ID'].shape)
            # print(new_g.dstdata['feat'].shape)
            # print(torch.max(new_g.dstdata['_ID']))

            # print(new_g)
            # print(new_g.all_edges('all'))
            cur_src_h = {}
            cur_dst_h = {}

            if len(new_g.srctypes) > 1:
                for k in new_g.srctypes:
                    if new_g.nodes[k].data['_ID'].size(0) == h[k].size(0):
                        cur_src_h[k] = h[k]
                        cur_src_h[k + '_src'] = h[k]
                    else:
                        select_ids = torch.cat(
                            [(g.nodes[k].data['_ID'] == i).nonzero() for i in _new_g.nodes[k].data['_ID']]).squeeze()
                        cur_src_h[k] = h[k][select_ids]
                        cur_src_h[k + '_src'] = h[k][select_ids]
            else:
                # print(new_g.srcdata['_ID'].size(0), h[new_g.srctypes[0]])
                if new_g.srcdata['_ID'].size(0) == h[new_g.srctypes[0]].size(0):
                    cur_src_h[new_g.srctypes[0]] = h[new_g.srctypes[0]]
                    cur_src_h[new_g.srctypes[0] + '_src'] = h[new_g.srctypes[0]]
                else:
                    select_ids = torch.cat(
                        [(g.nodes[new_g.srctypes[0]].data['_ID'] == i).nonzero() for i in
                         new_g.srcdata['_ID']]).squeeze()
                    cur_src_h[new_g.srctypes[0]] = h[new_g.srctypes[0]][select_ids]
                    cur_src_h[new_g.srctypes[0] + '_src'] = h[new_g.srctypes[0]][select_ids]

            if len(new_g.dsttypes) > 1:
                for k in new_g.dsttypes:
                    if new_g.nodes[k].data['_ID'].size(0) == h[k].size(0):
                        cur_dst_h[k] = h[k]
                        cur_dst_h[k + '_dst'] = h[k]
                    else:
                        select_ids = torch.cat([(g.nodes[k].data['_ID'] == i).nonzero() for i in
                                                new_g.nodes[k].data['_ID']]).squeeze()
                        cur_dst_h[k] = h[k][select_ids]
                        cur_dst_h[k + '_dst'] = h[k][select_ids]
            else:
                if new_g.dstdata['_ID'].size(0) == h[new_g.dsttypes[0]].size(0):
                    cur_dst_h[new_g.dsttypes[0]] = h[new_g.dsttypes[0]]
                    cur_dst_h[new_g.dsttypes[0] + '_dst'] = h[new_g.dsttypes[0]]
                else:
                    select_ids = torch.cat(
                        [(g.nodes[new_g.dsttypes[0]].data['_ID'] == i).nonzero() for i in new_g.dstdata['_ID']]).squeeze()
                    cur_dst_h[new_g.dsttypes[0]] = h[new_g.dsttypes[0]][select_ids]
                    cur_dst_h[new_g.dsttypes[0] + '_dst'] = h[new_g.dsttypes[0]][select_ids]

            # for k in cur_src_h.keys():
            #     print(k, cur_src_h[k].shape)
            # for k in cur_dst_h.keys():
            #     print(k, cur_dst_h[k].shape)

            cur_h = (cur_src_h, cur_dst_h)
            res = self.gat_layers[i](new_g, cur_h)

            for k in res.keys():
                semantic_embeddings[k].append(res[k].flatten(1))
                # self.semantic_attention(res[k])

        for k in semantic_embeddings.keys():
            semantic_embeddings[k] = torch.stack(semantic_embeddings[k], dim=1)
            res = self.semantic_attention(semantic_embeddings[k])
            # print(semantic_embeddings[k].shape)
            # print(res.shape)
            semantic_embeddings[k] = res
        return semantic_embeddings
