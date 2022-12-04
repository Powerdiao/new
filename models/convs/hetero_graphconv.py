# @Time     : 2021/5/2 22:49
# @Author   : Allen Xiong
# @File     : hetero_graphconv.py
import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn

class HeteroGraphConv(nn.Module):
    def __init__(self, g, in_feats, out_feats):
        '''

        :param g: HeteroGraph
        :param in_feats: dict, the feature size of inputs
        :param out_feats: dict, the output size
        '''
        super(HeteroGraphConv, self).__init__()
        self.Ws = nn.ModuleDict()
        for etype in g.canonical_etypes:
            utype, _, vtype = etype
            self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
        for ntype in g.ntypes:
            self.Ws[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])

    def forward(self, g, h):
        with g.local_scope():
            for ntype in g.ntypes:
                h_src, h_dst = h[ntype]
                g.dstnodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                g.srcnodes[ntype].data['h_src'] = h[ntype]
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                g.update_all(
                    dgl.function.copy_u('h_src', 'm'), dgl.function.mean('m', 'h_neigh'),
                    etype=etype)
                g.dstnodes[vtype].data['h_dst'] = \
                    g.dstnodes[vtype].data['h_dst'] + \
                    self.Ws[etype](g.dstnodes[vtype].data['h_neigh'])
            return {ntype: g.dstnodes[ntype].data['h_dst']
                    for ntype in g.ntypes}