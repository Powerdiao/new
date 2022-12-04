#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 14:19
# @Author  : Allen Xiong
# @File    : hg_sampler.py.py
import dgl
import torch
import numpy as np

from collections import defaultdict


class HGSampling(dgl.dataloading.BlockSampler):
    def __init__(self, num_layers, return_eids=False):
        super(HGSampling, self).__init__(num_layers, return_eids)

    def sample_subgraph(self, graph, seed_nodes, num_layers=2, sampled_number=8):
        # target node

        layer_data = defaultdict( # target type
            lambda : {} # target_id: [ser, time]
        )

        budget = defaultdict( # source type
            lambda : defaultdict( # source id
                float # sampled_score
            )
        )


        def add_budget(source_ids, source_ntype, budget):
            if len(source_ids) < sampled_number:
                sampled_ids = source_ids
            else:
                sampled_ids = source_ids[torch.randperm(len(source_ids))[:sampled_number]]

            for nid in sampled_ids:
                budget[source_ntype][nid] += 1. / len(sampled_ids)




        for ntype in graph.ntypes:
            for nid in graph.nodes(ntype):
                layer_data[ntype][nid] = len(layer_data[ntype])

        for etype in graph.canonical_etypes:
            source_ids = graph[etype].edges()[0]
            add_budget(source_ids, etype[0], budget)

        for layer in range(num_layers):
            source_ntypes = list(budget.keys())
            for source_ntype in source_ntypes:

                keys = np.array(list(budget[source_ntype].keys()))
                if(sampled_number > len(budget[source_ntype])):
                    sampled_ids = np.arange(len(budget[source_ntype]))
                else:
                    score = torch.tensor(list(budget[source_ntype].values())) ** 2
                    score = score / torch.sum(score)
                    sampled_ids = np.random.choice(len(score), sampled_number, p=score, replace=False)
                sampled_nodes = keys[sampled_ids]
                '''
                    First adding the sampled nodes then updating budget.
                '''
                for k in sampled_nodes:
                    layer_data[source_ntype][k] = len(layer_data[source_ntype])
                for k in sampled_nodes:
                    add_budget()


    def sample_frontier(self, block_id, g, seed_nodes):

        frontier = dgl.sampling.sample_neighbors(g, seed_nodes, -1, replace=self.replace)





