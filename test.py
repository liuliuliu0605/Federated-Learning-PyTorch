#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import datetime
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset_cluster, get_dataset,average_weights, mixing_weights, exp_details_cluster, get_device, init_seed,\
    complete, star, ring, independent, plot_label_dis, update_topo, optimal_mixing_cycles

from simulator.wsn.python.network.network import Network
from simulator.wsn.python.utils.utils import *
from simulator.wsn.python.routing.fcm import *
from statsmodels.stats.weightstats import DescrStatsW


fake = False

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)
    init_seed(args.seed)
    exp_details_cluster(args)
    device = get_device(args)


    # load dataset and user groups list
    train_dataset, test_dataset, user_groups_list = get_dataset_cluster(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
