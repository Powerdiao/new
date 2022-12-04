#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/27 22:50
# @Author  : Allen Xiong
# @File    : hgt_layer.py

import dgl
import torch
import torch.nn as nn
import numpy as np
import dgl.function as fn

from dgl.ops import edge_softmax


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_dict, edge_dict, num_heads, dropout=0.2, use_norm=True):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_ntypes = len(node_dict)
        self.num_etypes = len(edge_dict)
        self.total_rels = self.num_ntypes * self.num_etypes * self.num_ntypes
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_d_k = np.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None

        self.k_linears  = nn.ModuleList()
        self.q_linears  = nn.ModuleList()
        self.v_linears  = nn.ModuleList()
        self.a_linears  = nn.ModuleList()
        self.norms      = nn.ModuleList()

        for t in range(self.num_ntypes):
            self.k_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.q_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.v_linears.append(nn.Linear(self.in_dim, self.out_dim))
            self.a_linears.append(nn.Linear(self.in_dim, self.out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(self.out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_etypes, self.num_heads), requires_grad=True)
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_etypes, self.num_heads, self.d_k, self.d_k), requires_grad=True)
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_etypes, self.num_heads, self.d_k, self.d_k), requires_grad=True)
        self.skip           = nn.Parameter(torch.ones(self.num_ntypes), requires_grad=True)

        self.register_parameter("relation_pri", param=self.relation_pri)
        self.register_parameter("relation_att", param=self.relation_att)
        self.register_parameter("relation_msg", param=self.relation_msg)
        self.register_parameter("skip", param=self.skip)

        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, g, h):
        # if g.is_block:
        #     g = dgl.block_to_graph(g)
        h, t = h
        with g.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            # print(len(g.canonical_etypes))
            for srctype, etype, dsttype in g.canonical_etypes:
                subgraph = g[srctype, etype, dsttype]
                if subgraph.num_edges() == 0:
                    # print(srctype, etype, dsttype, " subgraph is empty", flush=True)
                    continue
                utype = srctype
                vtype = dsttype
                # utype = srctype[:-4]
                # vtype = dsttype[:-4]
                # print(srctype, dsttype)
                k_linear = self.k_linears[node_dict[utype]]
                q_linear = self.q_linears[node_dict[vtype]]
                v_linear = self.v_linears[node_dict[utype]]

                k = k_linear(h[utype]).view(-1, self.num_heads, self.d_k)
                q = q_linear(t[vtype]).view(-1, self.num_heads, self.d_k)
                v = v_linear(h[utype]).view(-1, self.num_heads, self.d_k)

                eids = self.edge_dict[(utype, etype, vtype)]

                relation_att = self.relation_att[eids]
                relation_pri = self.relation_pri[eids]
                relation_msg = self.relation_msg[eids]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)
                # print(h[vtype].shape)
                # print(torch.max(subgraph.nodes[dsttype].data['_ID']))
                # print(subgraph.nodes[dsttype].data['_ID'].shape)
                g[srctype, etype, dsttype].srcdata['k'] = k
                g[srctype, etype, dsttype].dstdata['q'] = q
                g[srctype, etype, dsttype].srcdata['v'] = v

                g[srctype, etype, dsttype].apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_scores = g[srctype, etype, dsttype].edata.pop('t').sum(dim=-1) * relation_pri / self.sqrt_d_k

                attn_scores = edge_softmax(g[srctype, etype, dsttype], attn_scores, norm_by='dst')
                g[srctype, etype, dsttype].edata['t'] = attn_scores.unsqueeze(-1)
                # print(subgraph.edata['t'].shape)
            # print(len(g.edata['t']))

            update_dict = {}
            for etype in g.canonical_etypes:
                if g[etype].num_edges() != 0:
                    # print('# ', etype, g[etype].dstdata.keys())
                    update_dict[etype] = (fn.u_mul_e('v', 't', 'm'), fn.sum('m', 'mu')) 

            # print(len(update_dict))
            g.multi_update_all(update_dict, cross_reducer='mean')
            
            # for dsttype in g.dsttypes:
            #     print(dsttype, g.nodes[dsttype].data.keys())
            # print(g.nodes['3'].data['label'].shape)
            # print(g.dsttypes)
            new_h = {}

            for srctype, etype, dsttype in g.canonical_etypes:
                subgraph = g[srctype, etype, dsttype]
                # print(srctype, etype, dsttype, subgraph.dstdata.keys())
                if subgraph.num_edges() == 0:
                    # print(srctype, etype, dsttype, " subgraph is empty", flush=True)
                    continue

                vtype = dsttype
                # vtype = ntype[:-4]

                nids = node_dict[vtype]
                alpha = torch.sigmoid(self.skip[nids])
                
                # print(dsttype, g.nodes[dsttype].data.keys())
                dst_t = g[srctype, etype, dsttype].dstdata['mu'].view(-1, self.out_dim)
                # print(ntype, dst_t)
                trans_out = self.dropout(self.a_linears[nids](dst_t))
                trans_out = trans_out * alpha + (1 - alpha) * t[vtype]
                if self.use_norm:
                    new_h[vtype] = self.norms[nids](trans_out)
                else:
                    new_h[vtype] = trans_out

            return new_h


