#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 15:07
# @Author  : Allen Xiong
# @File    : ACMDataset.py.py
import os
import torch
import dgl
import urllib.request
import scipy.io
import numpy as np

from dgl.data.utils import save_graphs, load_graphs
from collections import defaultdict

class ACMDataset(object):
    '''
    extract papers published in KDD, SIGMOD, SIGCOMM, MobiCOMM, and VLDB
    and divide the papers into three classes (Database, Wireless Communication, Data Mining)
    '''

    def __init__(self, force_reload=True, verbose=True):
        self.verbose = verbose
        self.force_reload = force_reload

        self.path = os.path.join(os.getenv("HOME"), ".dgl")

        self._feature_size = 128

        if(force_reload):
            data_path = os.path.join(os.getenv("HOME"), ".dgl/ACM.mat")
            data_path = os.path.abspath(data_path)
            web_url = 'https://data.dgl.ai/dataset/ACM.mat'
            if(not os.path.exists(data_path)):
                urllib.request.urlretrieve(web_url, data_path)

            data = scipy.io.loadmat(data_path)

            self._graph = dgl.heterograph({
                ('paper', 'written-by', 'author'): data['PvsA'].nonzero(),
                ('author', 'writing', 'paper'): data['PvsA'].transpose().nonzero(),
                ('paper', 'citing', 'paper'): data['PvsP'].nonzero(),
                ('paper', 'cited', 'paper'): data['PvsP'].transpose().nonzero(),
                ('paper', 'is-about', 'subject'): data['PvsL'].nonzero(),
                ('subject', 'has', 'paper'): data['PvsL'].transpose().nonzero(),
            })

            node_feats = {}
            edge_train_dicts = {}
            edge_valid_dicts = {}
            edge_test_dicts = {}


            for k in self._graph.ntypes:
                node_feats[k] = torch.randn(self._graph.num_nodes(k), self._feature_size)

            for etype in self._graph.canonical_etypes:
                edge_train_dicts[etype[1]] = np.random.rand(self._graph[etype].num_edges())
                edge_valid_dicts[etype[1]] = edge_test_dicts[etype[1]] = edge_train_dicts[etype[1]]

                edge_train_dicts[etype[1]] = edge_train_dicts[etype[1]] < 0.85
                edge_valid_dicts[etype[1]] = (edge_valid_dicts[etype[1]] < 0.95) & ~edge_train_dicts[etype[1]]
                edge_valid_dicts[etype[1]] = torch.tensor(edge_valid_dicts[etype[1]])
                edge_test_dicts[etype[1]] = torch.tensor(edge_test_dicts[etype[1]] >= 0.95)
                edge_train_dicts[etype[1]] = torch.tensor(edge_train_dicts[etype[1]])


            self._graph.ndata['feat'] = node_feats
            self._graph.edata['train_mask'] = edge_train_dicts
            self._graph.edata['valid_mask'] = edge_valid_dicts
            self._graph.edata['test_mask'] = edge_test_dicts

            # self._graph.ndata['_ID']
            # self._graph.edata['_ID']

            self.save()
        else:
            self.load()


        self._num_edges = self._graph.number_of_edges()
        self._num_rels = len(self._graph.etypes)
        self._num_node_type = len(self._graph.ntypes)

        self._print_info()

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

    def save(self):
        graph_path = os.path.join(self.path, 'ACM_dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.path, 'ACM_dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self._graph = graphs[0]

    def __getitem__(self, idx):
        assert idx == 0
        return self._graph

    def _print_info(self):
        if self.verbose:
            print('Finished data loading.')
            print('  NumNodes: {}'.format(self._graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self._graph.number_of_edges()))
            print('  NumFeats: {}'.format(self._feature_size))
            print('  NumNodesType: {}'.format(self.num_node_type))
            print('  NumRelsType: {}'.format(self.num_rels))
