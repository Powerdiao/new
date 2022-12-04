# @Time     : 2021/5/15 14:02
# @Author   : Allen Xiong
# @File     : hetero_rgcn_layer.py
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.function as fn


class HeterRGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, etypes):
        super(HeterRGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.ModuleDict({
            str(etype): nn.Linear(self.in_dim, self.out_dim)
            for etype in etypes
        })

        # self.weight = dglnn.HeteroGraphConv({
        #     etype[1]: dglnn.GraphConv(self.in_dim, self.out_dim, norm='right', weight=True, bias=True)
        #     for etype in etypes
        # })

    def forward(self, g, h):
        # return self.weight(g, h)

        if g.is_block:
            g = dgl.block_to_graph(g)
        with g.local_scope():
            funcs = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                canonical_etype = (srctype[:-4], etype, dsttype[:-4])

                Wh = self.weight[str(canonical_etype)](h[canonical_etype[0]])

                g.nodes[srctype].data['Wh_{}'.format(canonical_etype)] = Wh

                funcs[(srctype, etype, dsttype)] = (fn.copy_u('Wh_{}'.format(canonical_etype), 'm'), fn.mean('m', 'h'))


            g.multi_update_all(funcs, 'sum')

            return {ntype[:-4]: v for ntype, v in g.ndata['h'].items()}
            # return g.ndata['h']