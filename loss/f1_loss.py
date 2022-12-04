#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/16 17:16
# @Author  : Allen Xiong
# @File    : f1_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
import numpy as np


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2 or y_pred.ndim == 1
        assert y_true.ndim == 1

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)

        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.sigmoid(y_pred) #(y_pred, dim=0)
        # print(y_pred.shape)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        print(y_true)
        print(y_pred)
        # precs, recs, thrs = metrics.precision_recall_curve(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), pos_label=0)
        # precs = np.sum(precs)
        # recs = np.sum(recs)
        # f1s = 2 * precs * recs / (precs + recs)
        # print(f1s)

        print("F1 {:4.4f} | Precision {:4.4f} | Recall {:4.4f} | TP {:4.4f} | TN {:4.4f} | FP {:4.4f} | FN {:4.4f}".format(f1, precision,
                                                                                                              recall,
                                                                                                              tp, tn,
                                                                                                              fp, fn))
        return 1 - f1.mean()