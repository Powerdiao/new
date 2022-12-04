#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/30 14:43
# @Author  : Allen Xiong
# @File    : main_link_prediction.py
import time
import datetime
import logging
import argparse
import traceback
import utils
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from dgl.data.utils import generate_mask_tensor
from sampler import *
from loss import *


logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

stream_formatter = logging.Formatter('%(filename)s[line:%(lineno)d]: - %(message)s')

fh = logging.FileHandler('logs/%s.txt' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
fh.setLevel(logging.INFO)
fh.setFormatter(file_formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(stream_formatter)

logger.addHandler(ch)
logger.addHandler(fh)


#### Entry point
def run(proc_id, n_gpus, args, device, data):
    # Unpack data
    train_mask, valid_mask, test_mask, num_rels, feature_size, g = data

    rel_names = list(g.etypes)

    # train_eid_dict = {
    #     etype: g.edges(etype=etype, form='eid')
    #     for etype in g.canonical_etypes}

    # built-in dataset
    if g.ntypes[0] == '_N':
        nfeat = torch.tensor(utils.generate_features(g.num_nodes(), 50), dtype=torch.float32)
        in_feats = nfeat.shape[1]

        g.nodes['_N'].data['feat'] = nfeat
    # my dataset
    else:
        in_feats = feature_size
        for k in g.ntypes:
            g.nodes[k].data['feat'] = g.nodes[k].data['feat'].float()

    feats = g.ndata['feat']

    train_eids = utils.mask2index(train_mask)
    valid_eids = utils.mask2index(valid_mask)
    test_eids = utils.mask2index(test_mask)

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)

    # Create sampler
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(',')])
    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    sampler = MultiLayerDropoutSampler(p=args.dropout, num_layers=args.num_layers)
    # sampler = RandomWalkwithRestartSampler(args.num_layers, train_eids, restart_prob=0.3)

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eids, sampler,
        device=device,
        # g_sampling=dgl.edge_subgraph(g, train_eids, preserve_nodes=True),
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(args.num_negs),  # NegativeSampler(g, args.num_negs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        # pin_memory=True,
        num_workers=args.num_workers)

    # Define model and optimizer
    # model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    MODEL = utils.get_model_class(args.model)
    if args.model == "RGCN_Hetero_Entity_Classify" or args.model == "RGCN_Hetero_Link_Prediction":
        model = MODEL(g, in_feats, args.num_output,
                      num_bases=-1,
                      num_hidden_layers=args.num_layers - 2,
                      dropout=args.dropout,
                      device=device)
    elif args.model == "RGCN":
        model = MODEL(in_feats, args.num_hidden, args.num_output,
                      num_hidden_layers=args.num_layers - 2,
                      canonical_etypes=g.canonical_etypes,
                      device=device)
    elif args.model == "HetGNN":
        model = MODEL(g, feature_size, args.num_output,
                     num_hidden_layers=args.num_layers - 2,
                     dropout=args.dropout,
                     device=device)
    elif args.model == "HGMPAN":
        model = MODEL(g=g,
                      ntypes=g.ntypes,
                      etypes=g.canonical_etypes,
                      in_dim=feature_size,
                      hidden_dim=args.num_hidden,
                      out_dim=args.num_output,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      dropout=args.dropout,
                      use_norm=True)
    elif args.model == "HinSAGE":
        model = MODEL(g, feature_size, args.num_output,
                      num_hidden_layers=args.num_layers - 2,
                      dropout=args.dropout,
                      device=device)
    elif args.model == 'HAN':

        model = MODEL(meta_paths=[[('paper', 'written-by', 'author')],
                                  [('author', 'writing', 'paper')],
                                  [('paper', 'citing', 'paper')],
                                  [('paper', 'cited', 'paper')],
                                  [('paper', 'is-about', 'subject')],
                                  [('subject', 'has', 'paper')],
                                  ],
                      in_dim=feature_size,
                      hidden_dim=args.num_hidden,
                      out_dim=args.num_output,
                      ntypes=g.ntypes,
                      etypes=g.etypes,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
    elif args.model == "HGT":
        model = MODEL(g=g,
                      ntypes=g.ntypes,
                      etypes=g.canonical_etypes,
                      in_dim=feature_size,
                      hidden_dim=args.num_hidden,
                      out_dim=args.num_output,
                      num_layers=args.num_layers,
                      num_heads=args.num_heads,
                      dropout=args.dropout,
                      use_norm=True)
    else:
        model = MODEL(in_feats, args.num_hidden, args.num_hidden, args.margin, rel_names)

    encoder = model.to(device)
    if args.decoder == "DotPred":
        decoder = utils.get_predictor(args.decoder)(g.ntypes, args.num_output, device)
    else:
        decoder = utils.get_predictor(args.decoder)(g.canonical_etypes, args.num_output, device)
    decoder = decoder.to(device)
    Loss = utils.get_loss_fn(args.loss)
    loss_fcn = Loss()

    #criterion = nn.CrossEntropyLoss()#
    # criterion = F1_Loss()
    criterion = nn.MarginRankingLoss(args.margin, reduction='sum')
    optimizer = optim.AdamW([{'params': encoder.parameters()},
                            {'params': decoder.parameters()}],
                           lr=args.lr)

    logger.info('-' * 50)
    logger.info("Dataloader size: {0}".format(len(dataloader)))
    # logger.info('Dataloader sample graph: {0}'.format(dataloader.collator.g_sampling.num_edges()))
    logger.info("Train Edges: {0}".format(np.sum([len(v) for v in train_eids.values() if v.shape != torch.Size([]) ])))
    logger.info("Test Edges: {0}".format( np.sum([len(v) for v in test_eids.values() if v.shape != torch.Size([]) ] ) ) )
    logger.info("Valid Edges: {0}".format(np.sum([len(v) for v in valid_eids.values() if v.shape != torch.Size([]) ])) )

    # logger.info(encoder)
    # logger.info(decoder)

    # for srctype, e_type, dsttype in g.canonical_etypes:
    #     logger.info("(", srctype, e_type, dsttype, ") : ", g[srctype, e_type, dsttype].num_edges())

    logger.info('-' * 50)



    # Training loop

    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0
    best_eval_mrr = -1
    for epoch in range(args.num_epochs):
        tic = time.time()  # 返回当前时间戳（1970纪元后经过的浮点秒数）

        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        loss_avg = []
        f1_avg = []
        precision_avg = []
        recall_avg = []
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):

            d_step = time.time()

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            blocks = [block.to(device) for block in blocks]
            # logger.info(blocks[0].num_edges())
            # logger.info(pos_graph.num_edges())
            # logger.info(neg_graph.num_edges())
            # for block in blocks:
            #     logger.info(block)
            if '_N' in g.ntypes:
                batch_inputs = {'_N': blocks[0].srcdata['feat']}
            else:
                batch_inputs = (blocks[0].srcdata['feat']), blocks[0].dstdata['feat']

            # Compute loss and prediction
            # outs = model(blocks, batch_inputs)
            # pos_score = model.predict(pos_graph, outs)
            # neg_score = model.predict(neg_graph, outs)
            scores = torch.Tensor().to(device)
            labels = torch.Tensor().to(device)

            encoder.train()
            decoder.train()

            outs = encoder(blocks, batch_inputs)

            for k in outs.keys():
                select_ids = torch.cat([(input_nodes[k] == i).nonzero() for i in pos_graph.ndata['_ID'][k]]).squeeze()
                outs[k] = outs[k][select_ids]

            pos_score = decoder(pos_graph, outs)
            neg_score = decoder(neg_graph, outs)

            # logger.info(f"POS: {pos_score}")
            # logger.info(f"NEG: {neg_score}"")
            # logger.info(decoder.rel_embedding)
            # logger.info(decoder.rel_embedding['_no_relation'])
            for etype in pos_score.keys():
                idx = g.canonical_etypes.index(etype)+1
                no_rel_idx = 0

                scores = torch.cat([scores, pos_score[etype], neg_score[etype] ])
                pos_score_size, neg_score_size = pos_score[etype].shape[0], neg_score[etype].shape[0]
                labels = torch.cat(
                    [labels, torch.ones(pos_score_size).to(device), torch.zeros(neg_score_size).to(device)])

            # logger.info(scores)
            scores = torch.sigmoid(scores)
            # logger.info(scores)
            # loss = utils.margin_ranking_loss(pos_score, neg_score, args.margin)
            # loss = utils.cross_entropy_loss(pos_score, neg_score)
            # loss = criterion(scores, labels).to(device)
            # loss = criterion(pos_score, neg_score)
            # loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="sum")
            loss = utils.triple_loss(pos_score, neg_score)
            # loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logger.info(decoder.rel_embedding[('author', 'writing', 'paper')].grad)
            # logger.info(encoder.adapt_ws['author'].weight.grad)
            # logger.info(encoder.gcs[0].a_linears[0].weight.grad)
            # logger.info(encoder.gcs[0].relation_pri.grad)
            # logger.info(encoder.gcs[0].skip.grad)
            # logger.info(encoder.out['author'].weight.grad)

            loss_avg.append(loss.cpu().item())
            f1, precision, recall = utils.f1_loss(y_true=labels, y_pred=scores)
            f1_avg.append(f1)
            precision_avg.append(precision)
            recall_avg.append(recall)
            t = time.time()

            # logger.info("[LOSS] ", loss.cpu().item(), ", time(s):", t - tic_step, flush=True)

            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if (step+1) % args.log_every == 0:
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                logger.info('Epoch {:05d} | Step {:05d} | Loss {:4.4f} | F1 {:4.4f} {:4.4f} {:4.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step+1, loss.item(), np.mean(f1_avg), np.mean(precision_avg), np.mean(recall_avg), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()

        logger.info('Epoch Time(s): {:.4f}'.format(toc - tic))
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        logger.info('[Total] Epoch {:05d} | Step {:05d} | Loss {:4.4f} | F1 {:4.4f} {:4.4f} {:4.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                epoch, step, np.sum(loss_avg), np.mean(f1_avg), np.mean(precision_avg), np.mean(recall_avg), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]),
                np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))

        if (epoch + 1) % args.eval_every == 0:
            # eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)
            mrr, mr, hits_1, hits_50, hits_100, ndcg = utils.evaluate_link_prediction(encoder, decoder, g, valid_eids, args.batch_size, args.num_layers, args.num_eval_negs, device)
            logger.info('Eval Link Prediction valid MRR {:.4f} MR {:.4f} Hits@1 {:.4f} Hits@50 {:.4f} Hits@100 {:.4f} NDCG {:.4f}'.format(mrr, mr, hits_1, hits_50, hits_100, ndcg))

            # precision, recall, f1_score = utils._evaluate_link_classification(model, g, test_eids, args.batch_size, args.num_layers,
            #                                                      device)
            # logger.info('Eval Link Classification valid Precision {:.4f} Recall {:.4f} F1 {:.4f}'.format(precision, recall, f1_score))

            # TODO Model save to checkpoint
            if mrr > best_eval_mrr:
                best_eval_mrr = mrr
                state = {'epoch': epoch,
                         "encoder_state_dict": encoder.state_dict(),
                         "decoder_state_dict": decoder.state_dict(),
                         "optimizer": optimizer.state_dict(),
                        #  "loss_fn": loss_fcn,
                         "loss": loss.cpu().item()}
            # if eval_acc > best_eval_acc:
            #     best_eval_acc = eval_acc
            #     best_test_acc = test_acc
            #
            #     state = {'epoch': epoch,
            #              "model_state_dict": model.state_dict(),
            #              "optimizer": optimizer.state_dict(),
            #              "loss_fn": loss_fcn,
            #              "loss": loss}
            #     logger.info("Save model checkpoint...")
                torch.save(state, "/home/LAB/zengfr/xyf/SoftwareDefectPredictionByKG/checkpoints/{}_best.ckpt".format(args.model))
            # logger.info('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

        avg += toc - tic

    logger.info('Avg epoch time: {}'.format(avg / (epoch)))

def main(args, device):
    # load reddit data

    Dataset = utils.get_dataset_class(args.dataset)
    if args.dataset == "AMDataset" or args.dataset == 'ACMDataset':
        data = Dataset(force_reload=False, verbose=True)
    else:
        data = Dataset(reverse=False, force_reload=False, verbose=True)
    g = data[0]
    num_rels = len(g.canonical_etypes)
    if g.canonical_etypes[0] == ('_N', '_E', '_N'):
        eids = np.arange(g.number_of_edges())

        data_dict = {}

        edicts = utils.edges_dicts(eids, g.edata['etype'])
        for etype, eid in edicts.items():
            if etype < num_rels:
                data_dict[('_N', str(etype), '_N')] = (g.find_edges(eid))

        logger.info(len(data_dict))

        g = dgl.heterograph(data_dict)

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_mask = g.edata['train_mask']
    val_mask = g.edata['valid_mask']
    test_mask = g.edata['test_mask']
    feature_size = data.feature_size
    g.create_formats_()

    # Pack data
    data = train_mask, val_mask, test_mask, num_rels, feature_size, g

    if device == -1:
        run(0, 0, args, 'cpu', data)
    else:
        run(0, 0, args, device, data)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=int, default=3,
                           help="GPU for gpu trianing,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument("--model", type=str, default='GraphSAGE',
                           help="Trianing model name,")
    argparser.add_argument("--dataset", type=str, default='FB15k237Dataset',
                           help="Trianing dataset class")
    argparser.add_argument("--loss", type=str, default='CrossEntropyLoss',
                           help="Trianing loss class")

    argparser.add_argument("--decoder", type=str, default='DisMult',
                           help="Decoder triples to scores")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-output', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--num-negs', type=int, default=5)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--margin', type=float, default=10.0)
    argparser.add_argument('--dropout', type=float, default=0.4)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--num-heads', nargs='+', type=int, default=[4, 4, 1],
                           help="List of multi-heads")
    argparser.add_argument('--num-eval-negs', type=int, default=100,
                           help="eval's neg sample number")
    args = argparser.parse_args()

    device = args.gpu

    logger.info(args)
    
    main(args, device)
