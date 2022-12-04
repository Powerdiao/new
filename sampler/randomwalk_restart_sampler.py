# @Time     : 2021/5/4 21:50
# @Author   : Allen Xiong
# @File     : randomwalk_restart_sampler.py
import dgl
import torch


class RandomWalkwithRestartSampler(dgl.dataloading.BlockSampler):
    def __init__(self, num_layers, eids, restart_prob=0.4, return_eids=False):
        super(RandomWalkwithRestartSampler, self).__init__(num_layers, return_eids)
        self.restart_prob = restart_prob
        self.eids = eids

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        if isinstance(g, dgl.distributed.DistGraph):
            pass
        else:
            # frontier = dgl.in_subgraph(g, seed_nodes)
            # frontier = dgl.sampling.random_walk(g, seed_nodes, metapath=[], restart_prob=self.restart_prob)
            # sg = dgl.edge_subgraph(g, self.eids)
            frontier = dgl.in_subgraph(g, seed_nodes)
            # frontier = dgl.sampling.sample_neighbors(g, seed_nodes, -1, replace=False)

            # for k in seed_nodes.keys():
            #     print(k, seed_nodes[k].shape)
            # print(frontier.num_edges())
            # print('-'*60)
        return frontier


    def __len__(self):
        return self.num_layers



