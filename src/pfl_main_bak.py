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
    complete, star, ring, independent, plot_label_dis


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    init_seed(args.seed)
    exp_details_cluster(args)
    device = get_device(args)

    tag = 'Cluster[{}]_Sim[{:.2f}]_Topo[{}].pkl'.\
        format(args.num_clusters, args.cluster_similarity, args.topo)

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs/%s_%s/' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tag))

    # load dataset and user groups list
    train_dataset, test_dataset, user_groups_list = get_dataset_cluster(args)
    num_classes = len(set(train_dataset.targets))
    print("\nLabel distribution in clusters:")
    for c, user_groups in enumerate(user_groups_list):
        targets = np.array(train_dataset.targets)[np.array(list(user_groups.values())).flatten()].tolist()
        dis = [(i, targets.count(i)) for i in range(num_classes)]
        print("Cluster %d: " % c, dis)
        fig = plt.figure()
        plt.bar([i for i, _ in dis], [c for _, c in dis])
        logger.add_figure('Cluster/%d/Targets/distribution' % c, fig)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = [], [], []
    targets_groups = []
    for user_groups in user_groups_list:
        for u in user_groups:
            targets = set(np.array(train_dataset.targets)[user_groups[u]])
            targets_groups.append(targets)

    X = range(num_classes)
    Y = range(num_classes)
    X, Y = np.meshgrid(X, Y)
    x, y = X.ravel(), Y.ravel()
    Z = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            c = 0
            for targets in targets_groups:
                if set(targets) == set([i, j]):
                    c += 1
            Z[i,j] = c
    z = Z.ravel()

    # surf = ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap='viridis', edgecolor='none')
    # surf = ax.plot_surface(X,Y,Z, linewidth=0)
    # ax.plot_wireframe(X,Y,Z, rstride=10, cstride=0)
    # ax.zaxis.set_major_locator(LinearLocator(6))
    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1
    ax.bar3d(x, y, bottom, width, depth, z, shade=True)
    logger.add_figure('All/Users/distribution', figure=fig)

    # train_dataset, test_dataset, user_groups_all = get_dataset(args)
    # cluster_size = args.num_users // args.num_clusters
    # user_groups_list = []
    # for c in range(args.num_clusters):
    #     user_groups = {}
    #     for u in range(c*cluster_size, (c+1)*cluster_size):
    #         user_groups[u] = user_groups_all[u]
    #     user_groups_list.append(user_groups)
    #     targets = np.array(train_dataset.targets)[np.array(list(user_groups.values()),dtype=int).flatten()].tolist()
    #     print("Cluster %d: " % c, [(i, targets.count(i)) for i in range(10)])

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

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    num_clusters = args.num_clusters
    global_weights = global_model.state_dict()
    partial_weights_list = [copy.deepcopy(global_weights) for i in range(num_clusters)]

    # Build topo
    topo_matrix = None
    if args.topo == 'complete':
        topo_matrix = complete(num_clusters)
    elif args.topo == 'star':
        topo_matrix = star(num_clusters)
    elif args.topo == 'ring':
        topo_matrix = ring(num_clusters)
    elif args.topo == 'independent':
        topo_matrix = independent(num_clusters)
    else:
        raise NotImplementedError()

    print("\nTopo:")
    for row in topo_matrix:
        print(','.join(map(lambda x: "%.3f" % x, row)))

    # Training
    train_loss_list, train_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    test_loss_list, test_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    print_every = 2
    # val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # FL in each cluster
        for c, user_groups in enumerate(user_groups_list):
            print("Cluster %d is running..." % c)
            local_weights, local_losses = [], []

            # update global weights
            global_model.load_state_dict(partial_weights_list[c])

            global_model.train()
            my_users = list(user_groups_list[c].keys())
            m = max(int(args.frac * len(my_users)), 1)
            idxs_users = np.random.choice(my_users, m, replace=False)

            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            partial_weights_list[c] = average_weights(local_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss_list[c].append(loss_avg)
            logger.add_scalar('Cluster/%d/Train/loss' % c, loss_avg, global_step=epoch)

        # mixing between clusters
        partial_weights_list_mixing = []
        if args.mix_ep > 0 and (epoch+1) % args.mix_ep == 0:
            for i in range(num_clusters):
                partial_weights_list_mixing.append(mixing_weights(partial_weights_list, topo_matrix[:, i]))
            partial_weights_list = partial_weights_list_mixing

        # update global model (used for evaluation)
        global_weights = average_weights(partial_weights_list)
        global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        for c, user_groups in enumerate(user_groups_list):
            list_acc, list_loss = [], []
            global_model.eval()
            my_users = list(user_groups_list[c].keys())
            for u in my_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[u], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            # train_loss_list[c].append(sum(list_loss)/len(list_loss))
            # logger.add_scalar('[Cluster %d]Train/loss' % c, train_loss_list[c][-1], global_step=epoch)
            train_accuracy_list[c].append(sum(list_acc)/len(list_acc))
            logger.add_scalar('Cluster/%d/Train/acc' % c, train_accuracy_list[c][-1], global_step=epoch)

            # test_accuracy_list[c].append(sum(list_acc) / len(list_acc))
            # logger.add_scalar('[Cluster %d]Test/acc' % c, test_accuracy_list[c][-1], global_step=epoch)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss_list)[:,-1])}')
            print('Train Accuracy: {:.2f}% \n'.format(100*np.mean(np.array(train_accuracy_list)[:,-1])))
            # print('Test Accuracy: {:.2f}% \n'.format(100 * np.mean(np.array(test_accuracy_list)[:, -1])))
        logger.add_scalar('All/Train/loss', np.mean(np.array(train_loss_list)[:, -1]), global_step=epoch)
        logger.add_scalar('All/Train/acc', np.mean(np.array(train_accuracy_list)[:,-1]), global_step=epoch)
        # logger.add_scalar('[All]Test/acc', np.mean(np.array(test_accuracy_list)[:,-1]), global_step=epoch)

        # print global testing loss after every 'i' rounds
        test_accuracy, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Testing Stats after {epoch+1} global rounds:')
            print(f'Testing Loss : {test_loss}')
            print('Testing Accuracy: {:.2f}% \n'.format(100*test_accuracy))
            logger.add_scalar('All/Test/loss', test_loss, global_step=epoch)
            logger.add_scalar('All/Test/acc', test_accuracy, global_step=epoch)


    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*np.mean(np.array(train_accuracy_list)[:,-1])))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_Cluster[{}]_Sim[{:.2f}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs, args.num_clusters, args.cluster_similarity)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss_list, train_accuracy_list], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
