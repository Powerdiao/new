# @Time     : 2021/5/12 15:26
# @Author   : Allen Xiong
# @File     : RotatE.py
import dgl
import torch
import torch.nn as nn
import sys

from collections import defaultdict


class RotatE(nn.Module):
    def __init__(self, canonical_etypes, dim, device):
        super(RotatE, self).__init__()

        self.canonical_etypes = canonical_etypes
        self.num_rels = len(self.canonical_etypes)
        self.dim_e = dim
        self.dim_r = dim // 2
        self.device = device
        self.pi = nn.Parameter(torch.Tensor([3.14159265358979323846]), requires_grad=False).to(self.device)

        self.rel_embedding = {}#nn.ModuleDict()
        self.rel_embedding['_no_relation'] = nn.Parameter(torch.empty(1, self.dim_r, requires_grad=True).to(device))
        for k in self.canonical_etypes:
            self.rel_embedding[k] = nn.Parameter(torch.empty(1, self.dim_r, requires_grad=True).to(device))

        for k in self.rel_embedding.keys():
            self.register_parameter(str(k), param=self.rel_embedding[k])
            nn.init.xavier_normal_(self.rel_embedding[k])


    def forward(self, g, node_embeds, is_neg=False):

        with g.local_scope():
            re_node_embeds = {}
            im_node_embeds = {}

            for ntype in node_embeds:
                re, im = torch.chunk(node_embeds[ntype], 2, dim=-1)
                re_node_embeds[ntype] = re
                im_node_embeds[ntype] = im

            g.ndata['re'] = re_node_embeds
            g.ndata['im'] = im_node_embeds


            re_rels = {}
            im_rels = {}
            scores = {}

            for etype in g.canonical_etypes:
                re_rel, im_rel = self.phase_edges(etype, is_neg)
                re_rels[etype] = re_rel #torch.repeat_interleave(re_rel.view(1, -1), g[etype].num_edges(), dim=0) # re_rel.repeat(g[etype].num_edges(), 1)
                im_rels[etype] = im_rel #torch.repeat_interleave(im_rel.view(1, -1), g[etype].num_edges(), dim=0) # im_rel.repeat(g[etype].num_edges(), 1)

            for etype in g.canonical_etypes:
                u_ids, v_ids = g[etype].edges()

                re_u = g.ndata['re'][etype[0]][u_ids]
                re_v = g.ndata['re'][etype[-1]][v_ids]
                im_u = g.ndata['im'][etype[0]][u_ids]
                im_v = g.ndata['im'][etype[-1]][v_ids]
                # print(etype, re_u.shape)
                re_score = (re_u * re_rels[etype] - im_u * im_rels[etype]) - re_v
                im_score = (im_u * re_rels[etype] + re_u * im_rels[etype]) - im_v

                # g.apply_edges(dgl.function.u_mul_e('re', 're_rels', 're_s1'), etype=etype)
                # g.apply_edges(dgl.function.e_sub_v('re_s1', 're', 're_s11'), etype=etype)
                # g.apply_edges(dgl.function.u_mul_e('im', 'im_rels', 're_s2'), etype=etype)
                # g.apply_edges(dgl.function.e_sub_v('re_s2', 're', 're_s21'), etype=etype)
                # re_score = g[etype].edata['re_s11'] - g[etype].edata['re_s21']
                #
                # g.apply_edges(dgl.function.u_mul_e('im', 're_rels', 'im_s1'), etype=etype)
                # g.apply_edges(dgl.function.e_sub_v('im_s1', 'im', 'im_s11'), etype=etype)
                # g.apply_edges(dgl.function.u_mul_e('re', 'im_rels', 'im_s2'), etype=etype)
                # g.apply_edges(dgl.function.e_sub_v('im_s2', 'im', 'im_s21'), etype=etype)
                # im_score = g[etype].edata['im_s11'] - g[etype].edata['im_s21']

                # re_score = re_rel * g[etype].srcdata['re'] - im_rel * g[etype].srcdata['im']
                # im_score = re_rel * g[etype].srcdata['im'] + im_rel * g[etype].srcdata['re']
                #
                # re_score = re_score - g[etype].dstdata['re']
                # im_score = im_score - g[etype].dstdata['im']

                scores[etype] = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)


            # g.edata['re_rels'] = re_rels
            # g.edata['im_rels'] = im_rels




            return scores

    def phase_edges(self, etype, is_neg=False):
        if is_neg:
            r = self.rel_embedding["_no_relation"].squeeze()
            r = r / (self.pi)
            re_rel = torch.cos(r)
            im_rel = torch.sin(r)
            return re_rel, im_rel
        else:
            r = self.rel_embedding[etype].squeeze()
            r = r / (self.pi)
            re_rel = torch.cos(r)
            im_rel = torch.sin(r)
            return re_rel, im_rel


    def evaluate(self, src_embeds, dst_embeds, etype):
        # head mode
        re_src, im_src = torch.chunk(src_embeds, 2, dim=-1)
        re_dst, im_dst = torch.chunk(dst_embeds, 2, dim=-1)

        re_rel, im_rel = self.phase_edges(etype)

        re_score = re_rel * re_src - im_rel * im_src
        im_score = re_rel * im_src + im_rel * re_src

        re_score = torch.cat([re_score - re_d for re_d in re_dst], dim=0).view(-1, re_dst.size(0), re_dst.size(1))
        im_score = torch.cat([im_score - im_d for im_d in im_dst], dim=0).view(-1, im_dst.size(0), im_dst.size(1))
        score = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1).T
        return score