# @Time     : 2021/5/19 13:38
# @Author   : Allen Xiong
# @File     : DotPred.py
import dgl
import torch
import torch.nn as nn
import numpy as np

class DotPred(nn.Module):
    def __init__(self, ntypes, in_dim, device):
        super(DotPred, self).__init__()
        self.head_linear = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),

        )
        self.tail_linear = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )

        self.sqrt_hd = np.sqrt(in_dim)
        self.cache = None

    def forward(self, g, node_embeds, ):
        with g.local_scope():
            g.ndata['x'] = node_embeds

            scores = {}

            for srctype, etype, dsttype in g.canonical_etypes:
                # g.nodes[srctype].data['head'] = self.head_linear(g.nodes[srctype].data['x'])
                # g.nodes[dsttype].data['tail'] = self.tail_linear(g.nodes[dsttype].data['x'])
                #
                # g.apply_edges(dgl.function.u_dot_v('head', 'tail', 'score'), etype=(srctype, etype, dsttype))
                g.apply_edges(dgl.function.u_sub_v('x', 'x', 'score'), etype=(srctype, etype, dsttype))
                # print(g[(srctype, etype, dsttype)].edata['score'].shape)

                scores[(srctype, etype, dsttype)] = g[(srctype, etype, dsttype)].edata['score'].sum(dim=-1) / self.sqrt_hd

            return scores

    def evaluate(self, src_embeds, dst_embeds, etype):
        src_embeds = self.head_linear(src_embeds)
        dst_embeds = self.tail_linear(dst_embeds)
        scores = torch.matmul(dst_embeds, src_embeds.T)
        return scores
