#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--num_clusters', type=int, default=5,
                        help="number of clusters")
    parser.add_argument('--cluster_similarity', type=float, default=1.0,
                        help='the similarity between clusters (0~1)')
    parser.add_argument('--topo', type=str, default='complete',
                        help="topology of PSes")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')

    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    local_group.add_argument('--local_iter', type=int, default=0, help="the number of local iterations: E")

    # parser.add_argument('--local_ep', type=int, default=1,
    #                     help="the number of local epochs: E")
    parser.add_argument('--mix_ep', type=int, default=1,
                        help="the number of rounds before mixing: tau")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # gpu usage
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu', default=0, type=int, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
    gpu_group.add_argument('--data_parallel', action='store_true', help="To use \
                            data parallelism")
    gpu_group.add_argument('--local_rank', type=int, default=None, help="To use \
                            distributed data parallelism and will be automatically \
                            set by torch.distributed.launch")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--fake', action='store_true', help="fake training")
    parser.add_argument('--log_location', type=str, default='../logs', help="fake training")
    parser.add_argument('--estimate', action='store_true', help="whether to estimate parameters")
    args = parser.parse_args()
    return args
