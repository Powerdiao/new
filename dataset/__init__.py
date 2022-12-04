#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 13:33
# @Author  : Allen Xiong
# @File    : __init__.py

from .mydataset import MyDataset
from .ACMDataset import ACMDataset
from dgl.data import RedditDataset, FB15kDataset, FB15k237Dataset, AMDataset

__all__ = [
    'MyDataset',
    'ACMDataset',

    'RedditDataset',
    'FB15kDataset',
    'FB15k237Dataset',
    'AMDataset',
]