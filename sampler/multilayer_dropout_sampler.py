# @Time     : 2021/5/3 11:22
# @Author   : Allen Xiong
# @File     : multilayer_dropout_sampler.py
import dgl
import torch


class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
    def __init__(self, p, num_layers, return_eids=False):
        super(MultiLayerDropoutSampler, self).__init__(num_layers, return_eids)

        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # Get all inbound edges to `seed_nodes` 获取 `seed_nodes` 的所有入边
        sg = dgl.in_subgraph(g, seed_nodes)
        # DGL实现了多个可用于生成边界的函数。dgl.in_subgraph() 是一个生成子图的函数
        # 该子图包括初始图中的所有节点和指定节点的入边。图中其他的边都被删除。
        # seed_nodes为最后的输出节点
        # 用户可以将其用作沿所有入边传递消息的边界。

        new_edges_masks = {}
        # Iterate over all edge types
        for etype in sg.canonical_etypes:
            edge_mask = torch.zeros(sg.number_of_edges(etype))
            edge_mask.bernoulli_(self.p)  # 以某种概率将种子节点的入边随机剔除
            new_edges_masks[etype] = edge_mask.bool()

        # Return a new graph with the same nodes as the original graph as a
        # frontier
        frontier = dgl.edge_subgraph(sg, new_edges_masks, preserve_nodes=True)
        return frontier

    def __len__(self):
        return self.num_layers