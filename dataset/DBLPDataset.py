# @Time     : 2021/5/17 18:50
# @Author   : Allen Xiong
# @File     : DBLPDataset.py.py
import torch
import torch.nn as nn
import pickle
import dgl
import os

class DBLPDataset(object):
    def __init__(self, force_reload=True, verbose=True):
        download_dir = dgl.data.get_download_dir()

        with open(os.path.join(download_dir, "ACM/node_features.pkl"), "rb") as f:
            node_feats = pickle.load(f)
        with open(os.path.join(download_dir, "ACM/edges.pkl"), "rb") as f:
            edges = pickle.load(f)
        with open(os.path.join(download_dir, "ACM/labels.pkl"), "rb") as f:
            labels = pickle.load(f)

        self.graph = dgl.from_scipy(edges[0])
        print(node_feats.shape)
        print(type(edges[0]))
        # print(len(labels))
        print(self.graph)


if __name__ == "__main__":
    dataset = DBLPDataset()