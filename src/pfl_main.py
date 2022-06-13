#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import os
import copy
import json
import datetime, time
import numpy as np
from tensorboardX import SummaryWriter
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm

import config
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LR
from utils import get_dataset_cluster, average_weights, mixing_weights, exp_details_cluster, get_device, init_seed,\
    complete, star, ring, independent, lr_decay, \
    plot_label_dis, plot_user_dis


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)
    init_seed(args.seed)
    exp_details_cluster(args)
    device = get_device(args)

    # Define paths
    tag = 'Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'. \
        format(args.dataset, args.num_clusters, args.cluster_similarity, args.topo, args.mix_ep, args.lr, args.frac)
    log_location = args.log_location
    file_name = '%s_%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), tag)
    logger_dir = os.path.join(log_location, file_name)
    # logger_dir = '../logs/%s_%s/' % (datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), tag)
    logger = SummaryWriter(logger_dir)

    # Build topo
    num_clusters = args.num_clusters
    topo_matrix = None
    if args.topo == 'complete':
        topo_matrix = complete(num_clusters)
    elif args.topo == 'star':
        topo_matrix = star(num_clusters)
    elif args.topo == 'ring':
        topo_matrix = ring(num_clusters, centroids=config.centroids)
    elif args.topo == 'independent':
        topo_matrix = independent(num_clusters)
    else:
        exit('Error: unrecognized topology')

    spectral_norm = np.linalg.norm(topo_matrix - 1 / num_clusters, ord=2) ** 2
    p = 1 - spectral_norm
    print("\nTopo (p=%.5f):" % p)
    for row in topo_matrix:
        print(','.join(map(lambda x: "%.3f" % x, row)))

    # Load dataset and split users into clusters.
    # If config.users_groups is None, the number of users in each cluster is equal.
    # Otherwise, the size is the same with config.users_groups.
    train_dataset, test_dataset, split = get_dataset_cluster(args, users_groups=config.users_groups)
    user_groups_list = split['train'][0]
    test_idxs_list = split['test'][1]
    num_classes = args.num_classes
    print("\nLabel distribution in clusters:")
    for c, user_groups in enumerate(user_groups_list):
        targets = np.array(train_dataset.targets)[np.array(list(user_groups.values())).flatten()].tolist()
        dis = [(i, targets.count(i)) for i in range(num_classes)]
        my_users = [user for user in user_groups]
        print("-Cluster %d: " % c, dis)
        # print("%d users:" % len(my_users), my_users)
        fig1 = plot_label_dis([i for i, _ in dis], [c for _, c in dis])
        logger.add_figure('Cluster/%d/Label/distribution' % c, figure=fig1)  # label distribution in a cluster
    fig2 = plot_user_dis(train_dataset, user_groups_list, num_classes, args.num_users)  # all user distribution
    logger.add_figure('All/User/distribution', figure=fig2)

    # Build model
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
            # global_model = ResNet(args=args)
            # from torchvision import models
            # global_model =  models.resnet18(pretrained=False)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        print(len_in)
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    elif args.model == 'lr':
        # Logistic regression
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = LR(dim_in=len_in, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    total_params = sum([w.numel() for w in global_model.parameters()])
    print("num of params: ", total_params)
    # from torchsummary import summary
    # summary(global_model, input_size=(1, 28, 28))

    # Copy weights to each cluster.
    global_weights = global_model.state_dict()
    cluster_weights_list = [copy.deepcopy(global_weights) for i in range(num_clusters)]
    cluster_data_ratio_list = [None for _ in range(num_clusters)]

    # Observations in each cluster.
    if args.estimate:
        cluster_params_list = [None for _ in range(num_clusters)]
        cluster_gradients_list = [None for _ in range(num_clusters)]
        cluster_agg_delta_list = [0 for i in range(num_clusters)]
        cluster_mix_delta_list = [0 for i in range(num_clusters)]

    # Training and testing results
    train_loss_list, train_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    test_loss_list, test_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    global_params_list = []
    print_every = 2

    # Every communication round intra a cluster
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        lr = lr_decay(args.lr, epoch)
        logger.add_scalar('Observation/lr', lr, global_step=epoch)
        if args.local_ep > 0:
            K = args.local_ep
        else:
            K = len(train_dataset) / args.num_users / args.local_bs  # 100 is the number of users

        # Aggregation in each cluster
        for c, user_groups in enumerate(user_groups_list):
            local_weights, local_losses = [], []
            global_model.load_state_dict(cluster_weights_list[c])

            # train
            global_model.train()

            # select participating users
            my_users = list(user_groups.keys())
            m = max(int(args.frac * len(my_users)), 1)
            idxs_users = np.random.choice(my_users, m, replace=False)

            # train models and estimate params
            if args.estimate:
                params_groups = []
                init_grad_groups = []

            for idx in idxs_users:
                # initialize local model in each participating user
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)

                # local updates and estimate params
                rs_params = {} if args.estimate else None
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,
                                                     rt_params=rs_params, num_params=total_params)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(loss)

                if args.estimate:
                    params_groups.append(rs_params['estmt_params'])
                    init_grad_groups.append(rs_params['init_grad_flatten'])

            # aggregate models in a cluster
            agg_weights = average_weights(local_weights)
            cluster_weights_list[c] = agg_weights
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss_list[c].append(loss_avg)
            logger.add_scalar('Cluster/%d/Train/loss' % c, loss_avg, global_step=epoch)
            cluster_data_ratio_list[c] = len(idxs_users)

            # aggregate parameters in a cluster
            if args.estimate:
                cluster_gradients_list[c] = np.mean(init_grad_groups, axis=0)
                partial_params = np.mean(params_groups, axis=0)
                gamma = np.var(init_grad_groups, axis=0).sum() / total_params
                G = np.linalg.norm(cluster_gradients_list[c]) ** 2 / total_params
                L = 1  # TODO
                agg_weight_delta = 0
                for name, _ in global_model.named_parameters():
                    agg_weight_delta += (agg_weights[name] - cluster_weights_list[c][name]).norm() ** 2
                cluster_params_list[c] = np.concatenate([partial_params, [gamma, G, L]])
                cluster_agg_delta_list[c] = agg_weight_delta.item()

        if args.estimate:
            # Collect params from all clusters and obtain global params
            agg_params = DescrStatsW(cluster_params_list, weights=cluster_data_ratio_list)
            agg_gradients = DescrStatsW(cluster_gradients_list, weights=cluster_data_ratio_list)

            psi = agg_gradients.var.sum() / total_params  # TODO
            epsilon = np.linalg.norm(agg_gradients.mean) ** 2

            global_params = np.concatenate([agg_params.mean, [psi, epsilon]])
            global_params_list.append(global_params)

            print("Global estimated params:")
            for i, name in enumerate(['loss', 'sigma', 'gamma', 'G', 'L', 'psi', 'epsilon']):
                logger.add_scalar('Estimation/%s' % name, global_params[i], global_step=epoch)

            logger.add_scalar('Observation/agg', np.mean(cluster_agg_delta_list), global_step=epoch)

        # Conduct mixing operations every communication round, but the mixing results may not be
        # recorded according mixing cycles
        cluster_weights_list_mixing = []
        for i in range(num_clusters):
            cluster_weights_list_mixing.append(mixing_weights(cluster_weights_list, topo_matrix[:, i]))
            delta = 0

            if args.estimate:
                for name in cluster_weights_list_mixing[i]:
                    delta += (cluster_weights_list_mixing[i][name] - cluster_weights_list[i][name]).norm() ** 2
                cluster_mix_delta_list[i] = delta.item()

        # Record the mixing results according to mixing cycles
        if args.mix_ep > 0 and (epoch+1) % args.mix_ep == 0:
            cluster_weights_list = cluster_weights_list_mixing

        if args.estimate:
            logger.add_scalar('Observation/mix', np.mean(cluster_mix_delta_list), global_step=epoch)

        # Test after mixing
        for c in range(num_clusters):
            global_model.load_state_dict(cluster_weights_list_mixing[c])
            test_acc, test_loss = test_inference(args, global_model, test_dataset, test_idxs_list[c])
            test_accuracy_list[c].append(test_acc)
            test_loss_list[c].append(test_loss)

        # # Update global model (used for evaluation)
        # global_weights = average_weights(cluster_weights_list)
        # global_model.load_state_dict(global_weights)

        # Print global training loss after every 'i' rounds
        train_loss = DescrStatsW(np.array(train_loss_list)[:, -1], weights=cluster_data_ratio_list).mean
        test_loss = DescrStatsW(np.array(test_loss_list)[:, -1], weights=cluster_data_ratio_list).mean
        test_accuracy = DescrStatsW(np.array(test_accuracy_list)[:, -1], weights=cluster_data_ratio_list).mean

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {train_loss}')
            print(f' \nAvg Testing Stats after {epoch+1} global rounds:')
            print(f'Testing Accuracy: {100*test_accuracy}%')
            print(f'Testing Loss: {test_loss}')

        logger.add_scalar('All/Test/acc', test_accuracy, global_step=epoch)
        logger.add_scalar('All/Test/loss', test_loss, global_step=epoch)
        logger.add_scalar('All/Train/loss', train_loss, global_step=epoch)

        # Print global testing result after every 'i' rounds
        # test_accuracy, test_loss = test_inference(args, global_model, test_dataset)
        # test_accuracy_list.append(test_accuracy)
        # test_loss_list.append(test_loss)

        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Testing Stats after {epoch+1} global rounds:')
        #     print('Testing Accuracy: {:.2f}% \n'.format(100*test_accuracy))
        #     logger.add_scalar('All/Test/acc', test_accuracy, global_step=epoch)

    # # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    with open(os.path.join(logger_dir, 'convergence.json'), 'w') as f:
        rs = {
            'train_loss': DescrStatsW(train_loss_list, cluster_data_ratio_list).mean.tolist(),
            'test_loss': DescrStatsW(test_loss_list, cluster_data_ratio_list).mean.tolist(),
            'test_acc': DescrStatsW(test_accuracy_list, cluster_data_ratio_list).mean.tolist()
        }
        f.write(json.dumps(rs, indent=4))
        # json.dump(dict, f, indent=4)

    with open(os.path.join(logger_dir, 'user_groups_list.json'), 'w') as f:
        f.write(json.dumps([list(user_groups.keys()) for user_groups in user_groups_list]))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))