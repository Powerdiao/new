#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 19:37
# @Author  : Allen Xiong
# @File    : HANLayer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import GATConv

class HANLayer(nn.Module):
    