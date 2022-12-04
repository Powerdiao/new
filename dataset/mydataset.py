import numpy as np
import scipy.sparse as sp
import multiprocessing
import torch
import dgl
import os

from dgl.data.utils import save_graphs, load_graphs, generate_mask_tensor
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm

class MyDataset(object):
    '''
    Software defect prediction dataset that comes from apache org
    '''
    def __init__(self, path="/data/mydataset-1003/", force_reload=True, reverse=False, verbose=False):
        self.path = path
        self.force_reload = force_reload
        self.verbose = verbose
        self.reverse = reverse
        self._num_workers = 16

        self.process()

    def process(self):
        if self.force_reload:
            # 输入是一个邻接矩阵， 和结点的特征向量，label。 label[i]表示id = i的结点的label
            # load coo adj matrix from npz
            coo_adj = sp.load_npz(os.path.join(self.path, "graph.edge.npz"))
            coo_adj_src = coo_adj.row  # 40787683 * 1
            coo_adj_dst = coo_adj.col  # 40787683 * 1
            coo_adj_data = coo_adj.data  # 40787683 * 1

            edges_len = coo_adj_data.shape[0]
            bar_size = edges_len // self._num_workers

            # load node infotmation from npz
            graph_data = np.load(os.path.join(self.path, "graph.node.npz"))
            features = graph_data["features"]  # 1239463 * 128
            labels = graph_data["labels"]  # 1239463 * 1

            # dicts needed by heterograph
            hetero_graph_dicts = defaultdict(tuple)
            hetero_node_dicts = defaultdict(dict)
            # hetero_graph_dicts = multiprocessing.Manager().dict()
            edge_train_dicts = defaultdict(int)
            edge_valid_dicts = defaultdict(int)
            edge_test_dicts = defaultdict(int)
            edge_types = defaultdict(list)

            node_labels = defaultdict(list)
            node_feats = defaultdict(list)

            global construct_hetero_graph
            '''
            hetero graph construct function
            '''
            def construct_hetero_graph(idx):
                tmp_coo_adj_src, tmp_coo_adj_data, tmp_coo_adj_dst = coo_adj_src[idx::self._num_workers], coo_adj_data[idx::self._num_workers], coo_adj_dst[idx::self._num_workers]
                for idx, (src, etype, dst) in tqdm(enumerate(zip(tmp_coo_adj_src, tmp_coo_adj_data, tmp_coo_adj_dst)),
                                                   total=tmp_coo_adj_data.shape[0]):
                    src_ntype, dst_ntype = str(labels[src]), str(labels[dst])
                    if (src_ntype, str(etype), dst_ntype) not in hetero_graph_dicts:
                        hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)] = [], []

                    hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)][0].append(src)
                    hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)][1].append(dst)

                    edge_train_dicts[str(etype)] += 1
                    edge_valid_dicts[str(etype)] += 1
                    edge_test_dicts[str(etype)] += 1

                    edge_types[str(etype)].append(idx)

            '''
            joblib's Parallel with require="sharedmem" or backend='multiprocessing' attributes is slower because the iterable way
            by single element to run
            '''

            # out = Parallel(n_jobs=self._num_workers, require="sharedmem")(delayed(construct_hetero_graph)(i) for i in range(self._num_workers))
            '''
            multiprocessing has the same problem like joblib, the data sharing limit the speed
            '''
            # p_list = []
            #
            # for i in range(self._num_workers):
            #     p = multiprocessing.Process(target=construct_hetero_graph, args=(i, ))
            #     p_list.append(p)
            #     p.start()
            #
            # for p in p_list:
            #     p.join()

            '''
            traverse all the edge to build heterograph
            '''
            for idx, (src, etype, dst) in tqdm(enumerate(zip(coo_adj_src, coo_adj_data, coo_adj_dst)), total=coo_adj_data.shape[0]):
                src_ntype, dst_ntype = str(labels[src]), str(labels[dst])
                if (src_ntype, str(etype), dst_ntype) not in hetero_graph_dicts:
                    hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)] = [], []
                # src node and dst node reencoding
                if src not in hetero_node_dicts[src_ntype]:
                    hetero_node_dicts[src_ntype][src] = len(hetero_node_dicts[src_ntype])
                if dst not in hetero_node_dicts[dst_ntype]:
                    hetero_node_dicts[dst_ntype][dst] = len(hetero_node_dicts[dst_ntype])
                # add src and dst node to heterograph dict
                hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)][0].append( hetero_node_dicts[src_ntype][src] )
                hetero_graph_dicts[(src_ntype, str(etype), dst_ntype)][1].append( hetero_node_dicts[dst_ntype][dst] )

                # print(src, ' _ ', src_ntype, ' - ', etype, ' - ', dst, ' _ ', dst_ntype)

                # train, valid, and test mask generate
                edge_train_dicts[(src_ntype, str(etype), dst_ntype)] += 1
                edge_valid_dicts[(src_ntype, str(etype), dst_ntype)] += 1
                edge_test_dicts[(src_ntype, str(etype), dst_ntype)] += 1

                edge_types[(src_ntype, str(etype), dst_ntype)].append(idx)

            # for k in hetero_graph_dicts.keys():
            #     hetero_graph_dicts[k] = (np.array(hetero_graph_dicts[k][0]), np.array(hetero_graph_dicts[k][1]) )
            #     print(k)

            '''
            train, valid, and test mask generate
            '''
            for k in edge_train_dicts.keys():
                edge_train_dicts[k] = np.random.rand(edge_train_dicts[k])
                edge_valid_dicts[k] = edge_train_dicts[k]
                edge_test_dicts[k] = edge_train_dicts[k]

                edge_train_dicts[k] = edge_train_dicts[k] < 0.85
                edge_valid_dicts[k] = (edge_valid_dicts[k] < 0.95) & ~edge_train_dicts[k]
                edge_valid_dicts[k] = torch.tensor(edge_valid_dicts[k])
                edge_test_dicts[k] = torch.tensor(edge_test_dicts[k] >= 0.95)
                edge_train_dicts[k] = torch.tensor(edge_train_dicts[k])

                edge_types[k] = torch.tensor(edge_types[k])

            # for k in node_feats.keys():
            #     node_feats[k] = torch.rand(node_feats[k], self.feature_size)

            for idx, label in tqdm(enumerate(labels), total=labels.shape[0]):
                n_label = str(label)
                node_feats[n_label].append(features[idx])
                node_labels[n_label].append(idx)

            for k, v in node_feats.items():
                node_feats[k] = torch.tensor(node_feats[k]).float()

            node_dicts = {}
            for k, v in node_labels.items():
                node_labels[k] = torch.tensor(node_labels[k])
                node_dicts[k] = node_labels[k].shape[0]

            # num_nodes_dict: Explicitly specify the number of nodes for each node type in the graph.
            self._graph = dgl.heterograph(hetero_graph_dicts, num_nodes_dict=node_dicts)

            # for ntype in self._graph.ntypes:
            #     print("  {0}: {1} -- {2}".format(ntype, self._graph.number_of_nodes(ntype=ntype), self._graph.nodes(ntype)))
            # node_types = graph_data["node_types"]
            self._graph.ndata['feat'] = node_feats
            self._graph.ndata['label'] = node_labels

            self._graph.edata['etype'] = edge_types
            self._graph.edata['train_mask'] = edge_train_dicts
            self._graph.edata['valid_mask'] = edge_valid_dicts
            self._graph.edata['test_mask'] = edge_test_dicts

            self.save()
        else:
            self.load()

        self._num_edges = self._graph.number_of_edges()
        self._num_nodes = self._graph.number_of_nodes()
        self._num_node_type = len(self._graph.ndata['label'].keys())
        self._num_rels = len(self._graph.edata['etype'].keys())
        self._feature_size = self._graph.ndata['feat']['0'].shape[1]

        self._print_info()


    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._feature_size))
            print('  NumNodesType: {}'.format(self.num_node_type))
            print('  NumRelsType: {}'.format(self.num_rels))
            # print('  NumTrainingSamples: {}'.format(dgl.backend.nonzero_1d(self._graph.ndata['train_mask']).shape[0]))
            # print('  NumValidationSamples: {}'.format(dgl.backend.nonzero_1d(self._graph.ndata['val_mask']).shape[0]))
            # print('  NumTestSamples: {}'.format(dgl.backend.nonzero_1d(self._graph.ndata['test_mask']).shape[0]))

    @property
    def num_node_type(self):
        return self._num_node_type

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_edges(self):
        return self._num_edges
    @property
    def feature_size(self):
        return self._feature_size

    def load(self):
        graph_path = os.path.join(self.path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]
        # self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
        # self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
        # self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())

        # self._print_info()

    def save(self):
        graph_path = os.path.join(self.path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def __getitem__(self, idx):
        assert idx == 0
        return self._graph

    def _print_dict(self, dicts):
        print("Dict: ")
        for k, v in dicts.items():
            print("  {0}: {1}".format(k, v))


if __name__ == '__main__':
    dataset = MyDataset(force_reload=False, verbose=True)
