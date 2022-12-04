#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 17:15
# @Author  : Allen Xiong
# @File    : __init__.py
from .cross_entropy import CrossEntropyLoss
from .f1_loss import F1_Loss
__all__ = [
    'CrossEntropyLoss',
    'F1_Loss',
]