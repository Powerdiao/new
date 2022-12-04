#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 13:36
# @Author  : Allen Xiong
# @File    : main_heterograph.py.py
import time
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

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)

    # Create sampler
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(',')])
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eids, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(args.num_negs), #NegativeSampler(g, args.num_negs),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

    # Define model and optimizer
    # model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    MODEL = utils.get_model_class(args.model)
    if args.model == "RGCN_Hetero_Entity_Classify" or args.model == "RGCN_Hetero_Link_Prediction":
        model = MODEL(g, in_feats, num_rels+1,
                      num_bases=-1,
                      num_hidden_layers=args.num_layers-2,
                      dropout=args.dropout,
                      device=device)
    elif args.model == 'HAN':
        model = MODEL(meta_paths=[[('paper', 'written-by', 'author'), ('author', 'writing', 'paper')], ['citing', 'cited'], ['is-about', 'has']],
                      in_size=feature_size,
                      hidden_size=args.num_hidden,
                      out_size=num_rels+1,
                      ntypes=g.ntypes,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
    else:
        model = MODEL(in_feats, args.num_hidden, args.num_hidden, args.margin, rel_names)

    model = model.to(device)

    score_predictor = utils.get_predictor()(args.margin)

    Loss = utils.get_loss_fn(args.loss)
    loss_fcn = Loss()

    criterion = nn.CrossEntropyLoss()# nn.MarginRankingLoss(args.margin, reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('-' * 50)
    print("Dataloader size: ", len(dataloader))
    print('-' * 50)

    # Training loop

    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        loss_avg = []
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):

            d_step = time.time()

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.to(device) for block in blocks]
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
            pos_score, neg_score = model.totrain(pos_graph, neg_graph, blocks, batch_inputs)
            for etype in pos_score.keys():
                idx = g.canonical_etypes.index(etype)+1
                no_rel_idx = 0
                scores = torch.cat([scores, pos_score[etype], neg_score[etype]])
                labels = torch.cat([labels, torch.ones_like(pos_score[etype]), no_rel_idx * torch.ones_like(neg_score[etype])])
            # scores = torch.cat([pos_score, neg_score])
            # labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

            # loss = utils.margin_ranking_loss(pos_score, neg_score, args.margin)
            # loss = utils.cross_entropy_loss(pos_score, neg_score)
            # loss = criterion(pos_score, neg_score, torch.Tensor([-1]).to(device))
            # scores = F.sigmoid(scores)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_avg.append(loss.cpu().item())
            # print("[LOSS] ", loss.cpu().item())

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if (step+1) % args.log_every == 0:
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('[{}] Epoch {:05d} | Step {:05d} | Loss {:4.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                    proc_id, epoch, step+1, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()

        toc = time.time()

        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('[Total] Epoch {:05d} | Step {:05d} | Loss {:4.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                epoch, step, np.sum(loss_avg), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]),
                np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))

        if (epoch + 1) % args.eval_every == 0:
            # eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)
            mrr, mr, hits_1, hits_50, hits_100 = utils._evaluate_link_prediction(model, g, valid_eids, args.batch_size, args.num_layers,
                                                                 device)
            print('Eval Link Prediction valid MRR {:.4f} MR {:.4f} Hits@1 {:.4f} Hits@50 {:.4f} Hits@100 {:.4f}'.format(mrr, mr, hits_1,
                                                                                                        hits_50,
                                                                                                        hits_100))

            # precision, recall, f1_score = utils._evaluate_link_classification(model, g, test_eids, args.batch_size, args.num_layers,
            #                                                      device)
            # print('Eval Link Classification valid Precision {:.4f} Recall {:.4f} F1 {:.4f}'.format(precision, recall, f1_score))


            # if eval_acc > best_eval_acc:
            #     best_eval_acc = eval_acc
            #     best_test_acc = test_acc
            #
            #     state = {'epoch': epoch,
            #              "model_state_dict": model.state_dict(),
            #              "optimizer": optimizer.state_dict(),
            #              "loss_fn": loss_fcn,
            #              "loss": loss}
            #     print("Save model checkpoint...")
            #     torch.save(state, "/home/LAB/zengfr/xyf/dgl_kg/checkpoint/{}_best.ckpt".format(model.__name__))
            # print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

        avg += toc - tic

    print('Avg epoch time: {}'.format(avg / (epoch)))

def main(args, device):
    # load reddit data
    # data = RedditDataset(self_loop=False, verbose=True)
    # data = FB15k237Dataset(reverse=False)
    # data = dgl.data.CoraGraphDataset()

    # data = MyDataset(path="/home/LAB/zengfr/xyf/dataset/mydataset/", reload=False, verbose=True)
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

        print(len(data_dict))

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
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=5)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--margin', type=float, default=50.0)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--num-heads', type=list, default=[8],
                           help="List of multi-heads")
    args = argparser.parse_args()

    device = args.gpu

    main(args, device)
