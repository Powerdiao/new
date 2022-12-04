#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:38
# @Author  : Allen Xiong
# @File    : __init__.py
from .stochastic_twoLayer_rgcn import StochasticTwoLayerRGCN
from .hetero_rgcn_layer import HeterRGCNLayer
__all__ = [
    'StochasticTwoLayerRGCN',
    'HeterRGCNLayer',
]