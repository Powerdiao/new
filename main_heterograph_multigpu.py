#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/6 13:37
# @Author  : Allen Xiong
# @File    : main_heterograph_multigpu.py
import time
import argparse
import traceback
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import numpy as np

from _thread import start_new_thread


def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function


#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    train_mask, val_mask, test_mask, num_rels, g = data

    #########################################################################################

    rel_names = list(g.etypes)

    print(len(rel_names), g.num_edges())

    train_eid_dict = {
        etype: g.edges(etype=etype, form='eid')
        for etype in g.canonical_etypes}

    # test_size = int(len(eids) * 0.1)
    # train_size = int(len(eids) - test_size)
    #
    # test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    # train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    #
    # adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    # adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    # neg_u, neg_v = np.where(adj_neg != 0)
    #
    # neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    #
    # test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    #
    # train_g = dgl.remove_edges(g, eids[:test_size])
    #
    # train_pos_g = dgl.heterograph((train_pos_u, train_pos_v), num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    # train_neg_g = dgl.heterograph((train_neg_u, train_neg_v), num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    #
    # test_pos_g = dgl.heterograph((test_pos_u, test_pos_v), num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    # test_neg_g = dgl.heterograph((test_neg_u, test_neg_v), num_nodes_dict={ntype: g.num_nodes(ntype) for ntype in g.ntypes})


    ######################################################################################



    nfeat = torch.tensor(generate_features(g.num_nodes(), 50), dtype=torch.float32)
    in_feats = nfeat.shape[1]
    g.nodes['_N'].data['feat'] = nfeat


    train_nid = torch.LongTensor(np.nonzero(train_mask)).squeeze()
    val_nid = torch.LongTensor(np.nonzero(val_mask)).squeeze()
    test_nid = torch.LongTensor(np.nonzero(test_mask)).squeeze()

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)


    if n_gpus > 0:
        num_per_gpu = (train_seeds.shape[0] + n_gpus - 1) // n_gpus
        train_seeds = train_seeds[proc_id * num_per_gpu :
                                  (proc_id + 1) * num_per_gpu \
                                  if (proc_id + 1) * num_per_gpu < train_seeds.shape[0]
                                  else train_seeds.shape[0]]

    # Create sampler
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(',')])
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        negative_sampler=NegativeSampler(g, args.num_negs),#dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

    # Define model and optimizer
    # model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = RGCN_LP(in_feats, args.num_hidden, args.num_hidden, rel_names)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = CrossEntropyLoss()
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

        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):

            d_step = time.time()

            # print(pos_graph.num_edges())
            # print(neg_graph.num_edges())

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.to(device) for block in blocks]
            batch_inputs = {'_N': blocks[0].srcdata['feat']}
            # batch_inputs = {blocks[0].ndata['feat'], blocks[0].ndata['feat']}
            # print(blocks[0].ndata)
            # print(type(blocks[0]))
            # print(batch_inputs)
            # print(blocks[0].srcdata['_ID'].shape)
            # print(pos_graph.ndata['_ID'].shape)

            # Compute loss and prediction
            pos_score, neg_score = model(pos_graph, neg_graph, blocks, batch_inputs)
            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            # if step % args.log_every == 0:
            #     gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            #     print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
            #         proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()

            '''
            if step % args.eval_every == 0 and proc_id == 0:
                eval_acc, test_acc = evaluate(model, g, nfeat, labels, train_nid, val_nid, test_nid, device)
                print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc

                    state = {'epoch': epoch,
                             "model_state_dict": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "loss_fn": loss_fcn,
                             "loss": loss}
                    print("Save model checkpoint...")
                    th.save(state, "/home/LAB/zengfr/xyf/knowledge_representation/checkpoint/{}_best.ckpt".format(model.__name__))
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
            '''
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB'.format(
                    proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]),
                    np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
        if epoch >= 5:
            avg += toc - tic
        if n_gpus > 1:
            torch.distributed.barrier()

    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch)))


def main(args, devices):
    # load reddit data
    # data = RedditDataset(self_loop=False, verbose=True)
    # data = FB15k237Dataset(reverse=False)
    # data = dgl.data.CoraGraphDataset()

    # data = MyDataset(path="/home/LAB/zengfr/xyf/dataset/mydataset/", reload=False, verbose=True)

    g = data[0]
    num_rels = data.num_rels

    eids = np.arange(g.number_of_edges())
    edicts = edges_dicts(eids, g.edata['etype'])
    data_dict = {}

    for etype, eid in edicts.items():
        if etype < num_rels:
            data_dict[('_N', str(etype), '_N')] = (g.find_edges(eid))

    print(len(data_dict))

    train_mask = g.edata['train_mask']
    val_mask = g.edata['val_mask']
    test_mask = g.edata['test_mask']
    g = dgl.heterograph(data_dict)
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()
    # Pack data
    data = train_mask, val_mask, test_mask, num_rels, g

    n_gpus = len(devices)
    print("Number of gpu(s): ", n_gpus)
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], data)
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(run),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='0',
                           help="GPU, can be a list of gpus for multi-gpu trianing,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument("--model", type=str, default='0',
                           help="GPU, can be a list of gpus for multi-gpu trianing,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument("--dataset", type=str, default='0',
                           help="GPU, can be a list of gpus for multi-gpu trianing,"
                                " e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=1)
    argparser.add_argument('--neg-share', default=False, action='store_true',
                           help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))

    main(args, devices)