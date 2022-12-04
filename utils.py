#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 16:28
# @Author  : Allen Xiong
# @File    : utils.py
from re import sub
import sys
import dgl
import logging
import importlib
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import sklearn.linear_model as lm
import sklearn.metrics as skm

from tqdm import tqdm
from collections import defaultdict
from dataset import *
from loss import *
from models import *
from sampler import *
from models.predictors import *
from joblib import Parallel, delayed
from sklearn import metrics

logger = logging.getLogger()


def get_model_class(model_class):
    '''
    :param model_class: Name of graph model class
    :return: model class
    '''
    return getattr(sys.modules[__name__], model_class)

def get_dataset_class(dataset_class):
    '''
    :param dataset_class:  Name of dataset class
    :return: dataset class
    '''
    return getattr(sys.modules[__name__], dataset_class)

def get_loss_fn(loss_class):
    '''
    :param loss_class: Name of loss class
    :return: loss class
    '''
    return getattr(sys.modules[__name__], loss_class)

def get_predictor(predictor_class=None):

    if predictor_class is None:
        return getattr(sys.modules[__name__], "ScorePredictor")
    else:
        return getattr(sys.modules[__name__], predictor_class)

def edges_dicts(eids, etypes, reverse=False):
    edicts = defaultdict(list)
    for eid, etype in zip(eids, etypes):
        edicts[etype.item()].append(eid)

    return edicts

def generate_features(node_size, feature_size):
    np.random.seed(1234)
    return np.random.randn(node_size, feature_size)

def margin_ranking_loss(pos_score, neg_score, margin):
    loss = 0
    # an example hinge loss
    for etype, p_score in pos_score.items():
        if len(p_score) != 0:
            n = p_score.shape[0]
            loss += ( -(p_score.view(n, -1) - neg_score[etype].view(n, -1) ) + margin).clamp(min=0).mean()
    return loss

def triple_loss(pos_score, neg_score):
    loss = 0
    for etype in pos_score.keys():
        if len(pos_score[etype] > 0):
            n = pos_score[etype].shape[0]
            loss += -1 * (F.logsigmoid(pos_score[etype]).view(n, -1) + F.logsigmoid(-neg_score[etype]).view(n, -1) ).mean()
    return loss

def cross_entropy_loss(pos_score, neg_score):
    loss = 0 # torch.tensor([0.0], requires_grad=True)
    for etype, p_score in pos_score.items():
        try:
            if len(p_score) != 0 and len(neg_score[etype]) != 0:
                n = p_score.shape[0]
                loss = loss + F.binary_cross_entropy_with_logits(
                    neg_score[etype].view(n, -1).mean(dim=-1).view(n, -1),
                    p_score.view(n, -1)
                )
        except ValueError as VE:
            print("[POS] ", p_score.shape, len(p_score))
            print("[NEG] ", neg_score[etype].shape, len(neg_score[etype]))
            print(VE)
            exit(-1)
    return loss

def mask2index(mask):
    '''
    :param mask: dict, key is canonical_etype and value is tensor of bool whether contains this edges
    :return: dict, key is canonical_etype and value is tensor of index
    '''
    ret = {}
    for k in mask.keys():
        t = torch.sort(torch.nonzero(mask[k]).squeeze()).values
        if len(t.shape) > 0 and t.shape[0] > 0:
            ret[k] = t
    return ret

def eids2nids(g, eids):
    if isinstance(eids, dict):
        nids = defaultdict(set)
        for k in eids.keys():
            src, dst = g[k].edges()
            for u in src:
                nids[k[0]].add(u)
            for v in dst:
                nids[k[-1]].add(v)
        for k in nids:
            nids[k] = torch.tensor(list(nids[k]), device=g.device)
    return nids

def dcg_at_k(ranking, k):
    ranking = torch.tensor(ranking)[:k]
    if ranking.size(0) > 0:
        return ranking[0] + torch.sum(ranking[1:] / torch.log2(torch.arange(2, ranking.size(0) + 1).float()).to(ranking.device))
    return 0.

def ndcg_at_k(ranking, k):
    dcg_max = dcg_at_k(sorted(ranking, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(ranking, k) / dcg_max

def evaluate(model, g, predictor, batch_size, num_negs, train_eids, val_eids, test_eids, device):
    """
    Evaluate the model on the validation set specified by ``valid_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():

        if isinstance(model, SAGE):
            # pred = model.inference(g, nfeat, device)
            pass
        else:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.EdgeDataLoader(
                g, val_eids, sampler,
                batch_size=batch_size,
                negative_sampler=dgl.dataloading.negative_sampler.Uniform(num_negs),
                shuffle=False,
                drop_last=False,
                num_workers=0,
                device=device
            )

            hits_1 = 0
            hits_3 = 0
            hits_10 = 0
            MRR = 0
            MR = 0
            e_tot = 0
            for input_nodes, pos_graph, neg_graph, blocks in tqdm(dataloader, total=len(dataloader)):
                blocks = [block.to(device) for block in blocks]
                batch_inputs = (blocks[0].srcdata['feat']), blocks[0].dstdata['feat']
                outs = model(blocks, batch_inputs)
                pos_score = predictor(pos_graph, outs)
                neg_score = predictor(neg_graph, outs)
                e_tot += int(pos_graph.num_edges() + neg_graph.num_edges())

                for k in pos_score.keys():
                    # pos_score[k] = pos_score[k].to('cpu')
                    # neg_score[k] = neg_score[k].to('cpu').view(-1)
                    pos_score[k] = pos_score[k].view(-1)
                    neg_score[k] = neg_score[k].view(-1)

                    # scores = torch.topk(torch.cat([pos_score[k], neg_score[k]]), min(10, pos_score[k].size(0) + neg_score[k].size(0)))[1]
                    # e_tot += 1
                    #
                    # if scores.size(0) == 0:
                    #     continue
                    # if (scores > pos_score[k].size(0)).nonzero().size(0) != 0:
                    #     hits_10 += 1
                    # if (scores[:3] > pos_score[k].size(0)).nonzero().size(0) != 0:
                    #     hits_3 += 1
                    # if (scores[0] > pos_score[k].size(0)).nonzero().size(0) != 0:
                    #     hits_1 += 1

                    for i in range(pos_score[k].shape[0]):
                        pos_s = pos_score[k][i].view(-1)
                        neg_ss = neg_score[k][i*num_negs:i*num_negs + num_negs].view(-1)
                        scores = torch.topk(torch.cat([pos_s, neg_ss]), min(10, pos_s.size(0) + neg_ss.size(0)))[0]

                        ranking = (scores == pos_s).nonzero()[0].item()

                        if ranking < 1:
                            hits_1 += 1
                        if ranking < 3:
                            hits_3 += 1
                        if ranking < 10:
                            hits_10 += 1
                        MR += ranking + 1
                        MRR += 1.0/(ranking + 1)

            hits_1 /= e_tot
            hits_3 /= e_tot
            hits_10 /= e_tot
            MR /= e_tot
            MRR /= e_tot

    model.train()

    return MRR, MR, hits_1, hits_3, hits_10


def evaluate_link_prediction(encoder, decoder, g, eids, batch_size, num_layers, num_eval_negs, device=None):
    '''
    without negtive sample
    :param model:
    :param g:
    :param eids: test edge ids
    :param batch_size:
    :param num_layers:
    :param device:
    :return:
    '''
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        sub_g = dgl.edge_subgraph(g, eids, preserve_nodes=True, store_ids=True)
        # sub_g = sub_g.to(device)

        # for etype in eids.keys():
        #     eids[etype] = eids[etype].to(device)

        # for etype in sub_g.canonical_etypes:
        #     logger.info(eids[etype].shape)

        logger.info("eval subgraph edge number: {}".format(sub_g.num_edges()))
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
        
        dataloader = dgl.dataloading.EdgeDataLoader(
            g, eids, sampler,
            device=device,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(num_eval_negs),
            batch_size=batch_size//200,
            shuffle=True,
            drop_last=False,
            # pin_memory=True,
            num_workers=0
        )

        # y = {ntype: torch.zeros(
        #         sub_g.number_of_nodes(ntype),
        #         encoder.out_dim,
        #         dtype=torch.float,
        #         device=device
        #         )
        #     for ntype in g.ntypes}

        ranks = []
        hits_1 = []
        hits_5 = []
        hits_10 = []
        mrr = []
        mr = []
        ndcg = []

        for input_nodes, pos_graph, neg_graph, blocks in tqdm(dataloader):
            blocks = [block.to(device) for block in blocks]
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            if '_N' in g.ntypes:
                batch_inputs = {'_N': blocks[0].srcdata['feat']}
            else:
                batch_inputs = (blocks[0].srcdata['feat']), blocks[0].dstdata['feat']
            outs = encoder(blocks, batch_inputs)

            # for k in outs.keys():
            #     select_ids = torch.cat([(input_nodes[k] == i).nonzero() for i in pos_graph.ndata['_ID'][k]]).squeeze()
            #     outs[k] = outs[k][select_ids]

            pos_score = decoder(pos_graph, outs)
            neg_score = decoder(neg_graph, outs)

            for etype in pos_score.keys():
                for idx, pos_s in enumerate(pos_score[etype]):
                    ranks.append(torch.sum(pos_s < neg_score[etype][idx*num_eval_negs:idx*num_eval_negs + num_eval_negs]).cpu() + 1)
        np.save('ranks.npy', np.array(ranks))
        for r in ranks:
            hits_1.append(r <= 1)
            hits_5.append(r <= 50)
            hits_10.append(r <= 100)
            mrr.append(1.0 / r)
            mr.append(r)
            # ndcg.append(ndcg_at_k(torch.tensor(r), len(scores[k])))

        hits_1 = torch.tensor(hits_1).sum() / sub_g.number_of_edges()
        hits_5 = torch.tensor(hits_5).sum() / sub_g.number_of_edges()
        hits_10 = torch.tensor(hits_10).sum() / sub_g.number_of_edges()
        mrr = torch.tensor(mrr).sum() / sub_g.number_of_edges()
        mr = torch.tensor(mr).sum() / sub_g.number_of_edges()
        ndcg = 0

    return mrr, mr, hits_1, hits_5, hits_10, ndcg


def _evaluate_link_prediction(encoder, decoder, g, eids, batch_size, num_layers, device=None):
    '''
    without negtive sample
    :param model:
    :param g:
    :param eids: test edge ids
    :param batch_size:
    :param num_layers:
    :param device:
    :return:
    '''
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        sub_g = dgl.edge_subgraph(g, eids, preserve_nodes=True, store_ids=True)
        sub_g = sub_g.to(device)

        nids = eids2nids(sub_g, eids)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
            sub_g, nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=device
        )

        y = {ntype: torch.zeros(
                sub_g.number_of_nodes(ntype),
                encoder.out_dim,
                dtype=torch.float,
                device=device
                )
            for ntype in g.ntypes}

        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            blocks = [block.to(device) for block in blocks]
            batch_inputs = (blocks[0].srcdata['feat'], blocks[0].dstdata['feat'])
            outs = encoder(blocks, batch_inputs)

            for k in outs.keys():
                if(input_nodes[k].shape[0] == outs[k].shape[0]):
                    y[k][input_nodes[k]] = outs[k].float()
                elif(output_nodes[k].shape[0] == outs[k].shape[0]):
                    y[k][output_nodes[k]] = outs[k].float()

        scores = defaultdict(list)

        # head prediction
        with sub_g.local_scope():
            sub_g.ndata['x'] = y

            for etype in tqdm(sub_g.canonical_etypes, total=len(sub_g.canonical_etypes)):
                batch_sz = 10000
                n_batch = (sub_g[etype].number_of_edges() + batch_sz - 1) // batch_sz


                src = sub_g[etype].edges()[0]
                dst = sub_g[etype].edges()[1]
                # select_idx = torch.randperm(src.size(0))

                if etype[0] == etype[-1]:
                    embeds = sub_g[etype].ndata['x']
                else:
                    embeds = sub_g[etype].ndata['x'][etype[-1]]

                for idx in range(n_batch):
                    batch_start = idx * batch_sz
                    batch_end = min((idx+1) * batch_sz, sub_g[etype].number_of_edges())

                    Src_feats = sub_g[etype].srcdata['x'][src[batch_start: batch_end]]
                    # Dst_feats = sub_g[etype].dstdata['x'][dst[batch_start: batch_end]]
                    # print("[SRC] ", Src_feats.shape)
                    # print("[DST] ", embeds.shape)
                    # select_ids = torch.stack([torch.randperm(src.size(0)) for i in range(batch_start, batch_end)])
                    tmp = decoder.evaluate(embeds, Src_feats, etype)
                    # print(tmp[0][src[0]], tmp[0])
                    # WARNNING the relation of score
                    tmp = [torch.sum(score > score[n]) for (score, n) in zip(tmp, dst[batch_start: batch_end])]
                    scores[etype] += tmp

        hits_1 = []
        hits_50 = []
        hits_100 = []
        mrr = []
        mr = []
        ndcg = []
        for k in scores.keys():
            scores[k] = torch.tensor(scores[k], dtype=torch.float32, device=device)
            scores[k] += 1
            hits_1.append(torch.sum((scores[k] <= 1)))
            hits_50.append(torch.sum((scores[k] <= 50)))
            hits_100.append(torch.sum((scores[k] <= 100)))
            mrr.append(1.0 / scores[k])
            mr.append(scores[k])
            ndcg.append(ndcg_at_k(torch.tensor(scores[k]), len(scores[k])))

        hits_1 = torch.tensor(hits_1).sum() / sub_g.number_of_edges()
        hits_50 = torch.tensor(hits_50).sum() / sub_g.number_of_edges()
        hits_100 = torch.tensor(hits_100).sum() / sub_g.number_of_edges()
        mrr = torch.cat(mrr).sum() / sub_g.number_of_edges()
        mr = torch.cat(mr).sum() / sub_g.number_of_edges()
        ndcg = torch.tensor(ndcg).mean()

    torch.cuda.empty_cache()
    encoder.train()
    decoder.train()

    return mrr, mr, hits_1, hits_50, hits_100, ndcg

def _evaluate_link_classification(model, g, eids, batch_size, num_layers, device=None):
    model.eval()
    with torch.no_grad():

        sub_g = dgl.edge_subgraph(g, eids, preserve_nodes=True, store_ids=True)
        sub_g = sub_g.to(device)

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
            sub_g, sub_g.ndata['_ID'], sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            device=device
        )

        y = {ntype: torch.zeros(
                g.number_of_nodes(ntype),
                model.out_dim,
                dtype=torch.float,
                device=device
            )
            for ntype in g.ntypes}

        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            blocks = [block.to(device) for block in blocks]
            batch_inputs = (blocks[0].srcdata['feat'], blocks[0].dstdata['feat'])
            outs = model(blocks, batch_inputs)
            for k in outs.keys():
                y[k][output_nodes[k]] = outs[k].float()

        scores = defaultdict(list)

        # tail prediction
        with sub_g.local_scope():
            sub_g.ndata['x'] = y
            for etype in tqdm(sub_g.canonical_etypes, total=len(sub_g.canonical_etypes)):
                batch_sz = 10000
                n_batch = (sub_g[etype].number_of_edges() + batch_sz - 1) // batch_sz

                src = sub_g[etype].edges()[0]
                dst = sub_g[etype].edges()[1]

                if etype[0] == etype[-1]:
                    embeds = sub_g[etype].ndata['x']
                else:
                    embeds = sub_g[etype].ndata['x'][etype[-1]]

                for idx in range(n_batch):
                    batch_start = idx * batch_sz
                    batch_end = min((idx + 1) * batch_sz, sub_g[etype].number_of_edges())

                    # Src_feats = sub_g[etype].srcdata['x'][src[batch_start: batch_end]]
                    Dst_feats = sub_g[etype].dstdata['x'][dst[batch_start: batch_end]]
                    # print("[SRC] ", Src_feats.shape)
                    # print("[DST] ", embeds.shape)
                    for k in model.rel_embedding.keys():
                        tmp = torch.mul(model.rel_embedding[k].float(), embeds)
                        tmp = torch.matmul(Dst_feats, tmp.T)
                        # print(tmp.shape)
                        tmp = [torch.sum(score < score[n.item()]).item() for (score, n) in
                               zip(tmp, dst[batch_start: batch_end])]
                        scores[etype].append(tmp)

        precisions = []
        recalls = []
        f1_scores = []
        preds = []
        labels = []
        for k in scores.keys():
            scores[k] = torch.tensor(scores[k]).T
            score = torch.argmax(scores[k], dim=-1)
            label = [list(model.rel_embedding.keys()).index(k)] * scores[k].size(0)
            preds += score
            labels += label
            # prec = metrics.precision_score(labels, score, average='macro')
            # recall = metrics.recall_score(labels, score, average='macro')
            # f1 = metrics.f1_score(labels, score, average='macro')
            #
            # precisions.append(prec * scores[k].size(0))
            # recalls.append(recall * scores[k].size(0))
            # f1_scores.append(f1 * scores[k].size(0))


        precisions = metrics.precision_score(labels, preds, average='macro')
        recalls = metrics.recall_score(labels, preds, average='macro')
        f1_scores = metrics.f1_score(labels, preds, average='macro')

    torch.cuda.empty_cache()
    model.train()

    return precisions, recalls, f1_scores


def inference_node(model, g, batch_size, device):
    with torch.no_grad():
        nids = g.ndata['label']
        sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
        train_dataloader = dgl.dataloading.NodeDataLoader(
            g, nids, sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            device=device
        )
        result = defaultdict(list)
        for input_nodes, output_nodes, blocks in train_dataloader:
            inputs = blocks[0].srcdata['feat']
            outs = model(blocks, inputs)
            for k, v in outs.items():
                result[k].append(v)

        for k in result.keys():
            result[k] = torch.cat(result[k])

        return result

def inference_edge(model, g, eids, batch_size, device):
    with torch.no_grad():
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.EdgeDataLoader(
            g, eids, sampler,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(2),
            shuffle=False,
            drop_last=False,
            num_workers=4,
            device=device
        )
        result = defaultdict(list)
        for input_nodes, pos_graph, neg_graph, blocks in dataloader:
            batch_inputs = (blocks[0].srcdata['feat']), blocks[0].dstdata['feat']
            outs = model(blocks, batch_inputs)
            for k, v in outs.items():
                result[k].append(v)

        for k in result.keys():
            result[k] = torch.cat(result[k])

        return result

'''
def compute_acc(emb, labels, train_eids, val_eids, test_eids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_eids = train_eids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
'''

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
    # y_pred = torch.nn.functional.sigmoid(y_pred)
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    # print("Precision {:4.4f} | Recall {:4.4f} | TP {:4.4f} | TN {:4.4f} | FP {:4.4f} | FN {:4.4f}".format(precision, recall, tp, tn, fp, fn))
    f1 = 2* (precision*recall) / (precision + recall + epsilon)

    # logger.info(
    #     "F1 {:4.4f} | Precision {:4.4f} | Recall {:4.4f} | TP {:4.4f} | TN {:4.4f} | FP {:4.4f} | FN {:4.4f}".format(f1,
    #                                                                                                                  precision,
    #                                                                                                                  recall,
    #                                                                                                                  tp,
    #                                                                                                                  tn,
    #                                                                                                                  fp,
    #                                                                                                                  fn))
    if not is_training:
        f1 = f1.detach()
    return f1.cpu().item(), precision.cpu().item(), recall.cpu().item()
