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

    tag = 'Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}].pkl'.\
        format(args.num_clusters, args.cluster_similarity, args.topo, args.mix_ep)

    # define paths
    path_project = os.path.abspath('../..')
    logger_dir = '../logs/%s_%s/' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), tag)
    logger = SummaryWriter(logger_dir)

    # setup wsn networks
    traces = {}
    network = Network()
    routing_topology = 'FCM'
    routing_protocol_class = eval(routing_topology)
    network.routing_protocol = routing_protocol_class()
    network.routing_protocol.setup_phase(network, -1)
    users_groups = network.routing_protocol.users_groups

    # load dataset and user groups list
    train_dataset, test_dataset, user_groups_list = get_dataset_cluster(args, users_groups=users_groups)
    # num_classes = len(set(train_dataset.targets.numpy().tolist()))
    num_classes = 10
    print("\nLabel distribution in clusters:")
    for c, user_groups in enumerate(user_groups_list):
        targets = np.array(train_dataset.targets)[np.array(list(user_groups.values())).flatten()].tolist()
        dis = [(i, targets.count(i)) for i in range(num_classes)]
        my_users = [user for user in user_groups]
        print("-")
        print("Cluster %d: " % c, dis)
        print("%d users:" % len(my_users), my_users)
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
    total_params = sum([w.numel() for w in global_model.parameters()])
    print("num of params: ", total_params)

    # copy weights
    num_clusters = args.num_clusters
    global_weights = global_model.state_dict()
    partial_weights_list = [copy.deepcopy(global_weights) for i in range(num_clusters)]
    agg_delta_list = [0 for i in range(num_clusters)]
    mix_delta_list = [0 for i in range(num_clusters)]

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
    network.set_topo(topo_matrix)
    spectral_norm = np.linalg.norm(topo_matrix - 1 / num_clusters, ord=2) ** 2
    p = 1 - spectral_norm
    print("\nTopo (p=%.5f):" % p)
    for row in topo_matrix:
        print(','.join(map(lambda x: "%.3f" % x, row)))


    # Training
    train_loss_list, train_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    test_loss_list, test_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    print_every = 2
    # val_loss_pre, counter = 0, 0

    mix_counter, dynamic_tau = 1, 1
    intra_energy_list, inter_energy_list = [], []
    all_params_list = []

    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        print("Remain energy:", network.get_remaining_energy())
        if network.get_remaining_energy() <= 0:
            print("\n Terminated for energy exhausted!")
            break
        # if network.get_remaining_energy() <= 50:
        #     print("\n Terminated for exceeding energy budget!")
        #     break

        mix_counter -= 1

        # FL in each cluster
        params_clusters = [None for _ in range(num_clusters)]
        gradient_clusters = [None for _ in range(num_clusters)]
        weight_clusters = [None for _ in range(num_clusters)]
        total_num_participation = 0
        for c, user_groups in enumerate(user_groups_list):
            local_weights, local_losses = [], []

            # update global weights
            global_model.load_state_dict(partial_weights_list[c])
            global_model.train()

            # get alive users
            # note that the clusters sequence number in network is not constant
            # we modified FCM to stabilize the clustering results.
            my_users_total = list(user_groups.keys())
            my_users = [node.id for node in network.get_nodes_by_membership(c)]
            total, alive = len(my_users_total), len(my_users)
            print("-" * 40)
            if alive > 0:
                print("Cluster %d is running..." % c)
                print("alive: %d, total: %d, alive-ratio: %.2f" % (alive, total, alive/total))
            else:
                print("Cluster %d is terminated !" % c)

            # select participating users
            m = max(int(args.frac * total), 1)
            m = min(m, alive)
            idxs_users = np.random.choice(my_users, m, replace=False)

            if not fake:
                params_groups = []
                gradient_groups = []
                for idx in idxs_users:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], logger=logger)
                    # estimate params
                    # if args.mix_ep == 0:
                    if True:
                        params, gradient = local_model.estimate(model=copy.deepcopy(global_model),
                                                                num_params=total_params)
                        params_groups.append(params)
                        gradient_groups.append(gradient)

                    # local updates
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                if len(params_groups) > 0:
                    # aggregate params in a cluster
                    gamma = np.var(gradient_groups, axis=0).sum() / total_params
                    partial_params = np.mean(params_groups, axis=0)
                    params_clusters[c] = np.concatenate([partial_params, [gamma]])
                    gradient_clusters[c] = np.mean(gradient_groups, axis=0)
                    weight_clusters[c] = len(idxs_users)
                else:
                    params_clusters[c] = [0, 0, 0, 0]  # (L, G, sigma, gamma)
                    gradient_clusters[c] = np.zeros(total_params)
                    weight_clusters[c] = 0
                # logger.add_scalar('Cluster/%d/Estimation/L' % c, params_clusters[c][0], global_step=epoch)
                # logger.add_scalar('Cluster/%d/Estimation/G' % c, params_clusters[c][1], global_step=epoch)
                # logger.add_scalar('Cluster/%d/Estimation/sigma' % c, params_clusters[c][2], global_step=epoch)
                grad = np.linalg.norm(gradient_clusters[c])**2 / total_params
                logger.add_scalar('Cluster/%d/Estimation/grad^2' % c, grad, global_step=epoch)
                total_num_participation += len(idxs_users)

            else:
                for idx in idxs_users:
                    local_weights.append(global_weights)
                    local_losses.append(0)

            # update global weights
            if len(local_losses) > 0:
                agg_weights = average_weights(local_weights)
                delta = 0
                for name in agg_weights:
                    delta += (agg_weights[name] - partial_weights_list[c][name]).norm()**2
                agg_delta_list[c] = delta.item()
                partial_weights_list[c] = average_weights(local_weights)
                loss_avg = sum(local_losses) / len(local_losses)
                train_loss_list[c].append(loss_avg)
                logger.add_scalar('Cluster/%d/Train/loss' % c, loss_avg, global_step=epoch)
            else:
                train_loss_list[c].append(None)

        # # update mixing matrix
        alive_clusters = network.get_alive_clusters()
        topo_matrix = update_topo(topo_matrix, alive_clusters)
        spectral_norm = np.linalg.norm(topo_matrix - 1 / num_clusters, ord=2) ** 2
        p = 1 - spectral_norm
        logger.add_scalar('Estimation/p', p, global_step=epoch)

        # if args.mix_ep == 0:
        if True:
            # print(params_list)
            agg_params = DescrStatsW(params_clusters, weight_clusters)
            agg_gradients = DescrStatsW(gradient_clusters, weight_clusters)
            psi = agg_gradients.var.sum() / total_params
            epsilon = np.linalg.norm(agg_gradients.mean)**2
            all_params = np.concatenate([agg_params.mean, [psi, epsilon]])
            all_params_list.append(all_params)

            # params_avg = np.array(params_clusters).sum(axis=0)/total_num_participation
            print("Estimated params (all):")
            logger.add_scalar('Estimation/loss', all_params[0], global_step=epoch)
            logger.add_scalar('Estimation/L', all_params[1], global_step=epoch)
            logger.add_scalar('Estimation/G', all_params[2], global_step=epoch)
            logger.add_scalar('Estimation/sigma', all_params[3], global_step=epoch)
            logger.add_scalar('Estimation/gamma', all_params[4], global_step=epoch)
            logger.add_scalar('Estimation/psi', all_params[5], global_step=epoch)
            logger.add_scalar('Estimation/epsilon', all_params[6], global_step=epoch)


            loss, L, G, sigma, gamma, psi, epsilon = np.mean(all_params_list, axis=0)
            K, tau = 50, args.mix_ep
            init_loss = all_params_list[0][0]
            rho = sigma + 8*K*gamma + 8*K*psi + 8*K*G
            first_part = init_loss / ( args.lr * K * (epoch+1))
            second_part = 5 * args.lr**2 * K * L**2 * (1 + 1/(K-1))**K * (1 + 2/p) * (tau**2 / p - tau) * rho

            # logger.add_scalar('Estimation/first_part', first_part, global_step=epoch)
            # logger.add_scalar('Estimation/second_part', second_part, global_step=epoch)

            # calculate optimal mixing cycles
            lr = args.lr
            inter_cost = 1
            budget = 500
            opt_tau = optimal_mixing_cycles(L, G, sigma, gamma, psi, args.lr, K, p, init_loss, 1, budget)
            print("Optimial mixing cycles: ", opt_tau)
            logger.add_scalar('Estimation/tau', opt_tau, global_step=epoch)

            Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * rho
            first_part = init_loss * inter_cost / (lr * K * budget * tau ** 2)
            second_part = Gamma * (1 + 2 / p) * (2 * tau / p - 1)
            logger.add_scalar('Estimation/first_part', first_part, global_step=epoch)
            logger.add_scalar('Estimation/second_part', second_part, global_step=epoch)

        # mixing between clusters
        partial_weights_list_mixing = []

        # constant mixing
        if args.mix_ep > 0 and (epoch+1) % args.mix_ep == 0:
            network.activate_mix()
            for i in range(num_clusters):
                partial_weights_list_mixing.append(mixing_weights(partial_weights_list, topo_matrix[:, i]))
                delta = 0
                for name in partial_weights_list_mixing[i]:
                    delta += (partial_weights_list_mixing[i][name] - partial_weights_list[i][name]).norm() ** 2
                mix_delta_list[i] = delta.item()
            logger.add_scalar('Observation/agg', np.mean(agg_delta_list), global_step=epoch)
            logger.add_scalar('Observation/mix', np.mean(mix_delta_list), global_step=epoch)
            partial_weights_list = partial_weights_list_mixing
        # dynamic mixing
        elif args.mix_ep == 0 and mix_counter <= 0:
            network.activate_mix()

            for i in range(num_clusters):
                partial_weights_list_mixing.append(mixing_weights(partial_weights_list, topo_matrix[:, i]))
                delta = 0
                for name in partial_weights_list_mixing[i]:
                    delta += (partial_weights_list_mixing[i][name] - partial_weights_list[i][name]).norm() ** 2
                mix_delta_list[i] = delta.item()
            logger.add_scalar('Observation/agg', np.mean(agg_delta_list), global_step=epoch)
            logger.add_scalar('Observation/mix', np.mean(mix_delta_list), global_step=epoch)
            if np.mean(agg_delta_list) < np.mean(mix_delta_list):
                dynamic_tau -= 1
                dynamic_tau = max(dynamic_tau, 1)
            else:
                dynamic_tau += 1

            partial_weights_list = partial_weights_list_mixing
            inter_energy_list.append(network.energy_dis['inter-comm'])
            mix_counter = dynamic_tau
            logger.add_scalar('Observation/tau', dynamic_tau, global_step=epoch)
        else:
            network.deactivate_mix()

        # update global model (used for evaluation)
        global_weights = average_weights(partial_weights_list)
        global_model.load_state_dict(global_weights)

        # # Calculate avg training result over all users at every epoch
        # for c, user_groups in enumerate(user_groups_list):
        #     list_acc, list_loss = [], []
        #     global_model.eval()
        #     my_users = list(user_groups_list[c].keys())
        #     for u in my_users:
        #         local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                                   idxs=user_groups[u], logger=logger)
        #         acc, loss = local_model.inference(model=global_model)
        #         list_acc.append(acc)
        #         list_loss.append(loss)
        #     # train_loss_list[c].append(sum(list_loss)/len(list_loss))
        #     # logger.add_scalar('[Cluster %d]Train/loss' % c, train_loss_list[c][-1], global_step=epoch)
        #     train_accuracy_list[c].append(sum(list_acc)/len(list_acc))
        #     logger.add_scalar('Cluster/%d/Train/acc' % c, train_accuracy_list[c][-1], global_step=epoch)
        #
        #     # test_accuracy_list[c].append(sum(list_acc) / len(list_acc))
        #     # logger.add_scalar('[Cluster %d]Test/acc' % c, test_accuracy_list[c][-1], global_step=epoch)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            losses_group = np.array(train_loss_list)[:,-1]
            print(f'Training Loss : {losses_group[losses_group != None].mean()}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*np.mean(np.array(train_accuracy_list)[:,-1])))
            # print('Test Accuracy: {:.2f}% \n'.format(100 * np.mean(np.array(test_accuracy_list)[:, -1])))
        losses_group = np.array(train_loss_list)[:, -1]
        logger.add_scalar('All/Train/loss', losses_group[losses_group != None].mean(), global_step=epoch)
        # logger.add_scalar('All/Train/acc', np.mean(np.array(train_accuracy_list)[:,-1]), global_step=epoch)

        # print global testing result after every 'i' rounds
        if not fake:
            test_accuracy, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy_list.append(test_accuracy)
            test_loss_list.append(test_loss)
        else:
            test_accuracy, test_loss = 0, 0
            test_accuracy_list.append(0)
            test_loss_list.append(0)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Testing Stats after {epoch+1} global rounds:')
            # print(f'Testing Loss : {test_loss}')
            print('Testing Accuracy: {:.2f}% \n'.format(100*test_accuracy))
            # logger.add_scalar('All/Test/loss', test_loss, global_step=epoch)
            logger.add_scalar('All/Test/acc', test_accuracy, global_step=epoch)

        # simulate one round in wsn
        traces[routing_topology] = network.simulate_one_round()

        intra_energy_list.append(network.energy_dis['intra-comm'] + network.energy_dis['local-update'])
        inter_energy_list.append(network.energy_dis['inter-comm'])

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*np.mean(np.array(train_accuracy_list)[:,-1])))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_Cluster[{}]_Sim[{:.2f}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.num_clusters, args.cluster_similarity)

    with open(os.path.join(logger_dir, 'convergence.pkl'), 'wb') as f:
        pickle.dump([train_loss_list, test_accuracy_list], f)

    # plot cluster information
    plot_clusters(network, logger_dir)
    plot_time_of_death(network, logger_dir)
    plot_traces(traces, logger_dir)
    print(traces[routing_topology]['alive_clusters'])

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
