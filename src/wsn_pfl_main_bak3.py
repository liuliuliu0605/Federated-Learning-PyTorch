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
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LR
from utils import get_dataset_cluster, get_dataset,average_weights, mixing_weights, exp_details_cluster, get_device, init_seed,\
    complete, star, ring, independent, plot_user_dis, plot_label_dis, update_topo, optimal_mixing_cycles, lr_decay

from simulator.wsn.python.network.network import Network
from simulator.wsn.python.utils.utils import *
from simulator.wsn.python.routing.fcm import *
from statsmodels.stats.weightstats import DescrStatsW


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)
    init_seed(args.seed)
    exp_details_cluster(args)
    device = get_device(args)
    fake = args.fake

    # define paths
    tag = 'Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'. \
        format(args.dataset, args.num_clusters, args.cluster_similarity, args.topo, args.mix_ep, args.lr, args.frac)
    logger_dir = '../logs/%s_%s/' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '[Energy]_%s' % tag)
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
    train_dataset, test_dataset, split = get_dataset_cluster(args, users_groups=users_groups)
    user_groups_list = split['train'][0]
    # num_classes = len(set(train_dataset.targets.numpy().tolist()))
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
    print(global_model)
    total_params = sum([w.numel() for w in global_model.parameters()])
    print("num of params: ", total_params)

    # copy weights
    num_clusters = args.num_clusters
    global_weights = global_model.state_dict()
    cluster_wegihts_list = [copy.deepcopy(global_weights) for i in range(num_clusters)]
    cluster_params_list = [None for _ in range(num_clusters)]
    cluster_gradients_list = [None for _ in range(num_clusters)]
    cluster_data_ratio_list = [None for _ in range(num_clusters)]
    cluster_agg_delta_list = [0 for i in range(num_clusters)]
    cluster_mix_delta_list = [0 for i in range(num_clusters)]

    # Build topo
    topo_matrix = None
    if args.topo == 'complete':
        topo_matrix = complete(num_clusters)
    elif args.topo == 'star':
        topo_matrix = star(num_clusters)
    elif args.topo == 'ring':
        centroids = [(node.pos_x, node.pos_y) for node in network.centroids]
        print(centroids)
        exit()
        topo_matrix = ring(num_clusters, centroids=centroids)
    elif args.topo == 'independent':
        topo_matrix = independent(num_clusters)
    else:
        exit('Error: unrecognized topology')

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

    opt_tau, mixing_timer, dynamic_tau, num_mixing = 1, 1, 1, 0
    intra_energy_list, inter_energy_list = [], []
    global_params_list = []
    budget = 50
    total_energy = network.get_remaining_energy()
    # local_weights_last_local_last_round = {}


    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        print("Remain energy:", network.get_remaining_energy())
        logger.add_scalar('Observation/energy_consumption', total_energy-network.get_remaining_energy(), global_step=epoch)
        if network.get_remaining_energy() <= total_energy-budget:
            print("\n Terminated for energy exhausted!")
            break

        # if network.get_remaining_energy() <= 50:
        #     print("\n Terminated for exceeding energy budget!")
        #     break

        lr = lr_decay(args.lr, epoch)
        K = len(train_dataset) / 100 / args.local_bs  # 100 is the number of users

        # used in dynamic mixing when decreasing to 0
        mixing_timer -= 1

        # FL in each cluster
        total_num_participation = 0

        for c, user_groups in enumerate(user_groups_list):
            local_weights, local_losses = [], []

            # load cluster model
            global_model.load_state_dict(cluster_wegihts_list[c])
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

            # select participating users, which should be alive
            m = max(int(args.frac * total), 1)
            m = min(m, alive)
            idxs_users = np.random.choice(my_users, m, replace=False)

            # whether fake training
            if not fake:
                params_groups = []
                gradient_groups = []
                weight_delta = 0
                for idx in idxs_users:
                    local_model = LocalUpdate(args=args, dataset=train_dataset,
                                              idxs=user_groups[idx], logger=logger)
                                              # last_local_weights = local_weights_last_local_last_round.get(idx, None))

                    # local updates and estimate params
                    # if args.mix_ep == 0:
                    rs_params = {}
                    w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,
                                                         rt_params=rs_params, num_params=total_params)
                    # local_weights_last_local_last_round[idx] = w
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                    # if args.mix_ep == 0:
                    params_groups.append(rs_params['estmt_params'])
                    gradient_groups.append(rs_params['init_grad_vector'])
                    weight_delta += rs_params['weight_delta']

                weight_delta /= len(idxs_users)

                # aggregate params in a cluster, which should have alive clients
                if len(params_groups) > 0:
                    # my_gradients_last_round = cluster_gradients_list[c]
                    cluster_gradients_list[c] = np.mean(gradient_groups, axis=0)
                    cluster_data_ratio_list[c] = len(idxs_users)
                    G = np.linalg.norm(cluster_gradients_list[c]) ** 2 / total_params
                    gamma = np.var(gradient_groups, axis=0).sum() / total_params
                    partial_params = np.mean(params_groups, axis=0)
                    # cluster_params_list[c] = np.concatenate([partial_params, [gamma]])
                    cluster_params_list[c] = np.concatenate([partial_params, [gamma, G]])

                else:
                    cluster_params_list[c] = [0, 0, 0, 0]  # (loss, sigma, gamma, G)
                    cluster_gradients_list[c] = np.zeros(total_params)
                    cluster_data_ratio_list[c] = 0

                grad_norm = np.linalg.norm(cluster_gradients_list[c])**2 / total_params
                total_num_participation += len(idxs_users)

                # logger.add_scalar('Cluster/%d/Estimation/grad^2' % c, grad_norm, global_step=epoch)
                # logger.add_scalar('Cluster/%d/Estimation/L' % c, cluster_params_list[c][0], global_step=epoch)
                # logger.add_scalar('Cluster/%d/Estimation/G' % c, cluster_params_list[c][1], global_step=epoch)
                # logger.add_scalar('Cluster/%d/Estimation/sigma' % c, cluster_params_list[c][2], global_step=epoch)

            else:
                # fake
                for idx in idxs_users:
                    local_weights.append(global_weights)
                    local_losses.append(0)

            # update cluster weights in case that energy exhausted
            if len(local_losses) > 0:
                agg_weights = average_weights(local_weights)

                grad_delta = 0
                index = 0
                for name in agg_weights:
                    grad_delta += np.linalg.norm((cluster_wegihts_list[c][name] - agg_weights[name]).cpu().numpy().reshape(-1) / lr
                                   - K * cluster_gradients_list[c][index:index+agg_weights[name].numel()]) ** 2
                    index += agg_weights[name].numel()

                L = np.sqrt(grad_delta/weight_delta/K)
                cluster_params_list[c] = np.concatenate([cluster_params_list[c], [L]])

                delta = 0
                for name in agg_weights:
                    delta += (agg_weights[name] - cluster_wegihts_list[c][name]).norm() ** 2
                cluster_agg_delta_list[c] = delta.item()

                my_weights_last_round = cluster_wegihts_list[c]
                cluster_wegihts_list[c] = agg_weights
                loss_avg = sum(local_losses) / len(local_losses)
                train_loss_list[c].append(loss_avg)
                logger.add_scalar('Cluster/%d/Train/loss' % c, loss_avg, global_step=epoch)
            else:
                train_loss_list[c].append(None)
                cluster_params_list[c] = np.concatenate([cluster_params_list[c], [0]])

            # agg gradients to calculate L, failed
            # if my_weights_last_round is not None and my_gradients_last_round is not None:
            #     weight_delta = 0
            #     for key in my_weights_last_round.keys():
            #         weight_delta += (my_weights_last_round[key] - cluster_wegihts_list[c][key]).norm().item() ** 2
            #     grad_delta = np.linalg.norm(my_gradients_last_round - cluster_gradients_list[c]) **2
            #     cluster_params_list[c][1] = np.sqrt(grad_delta / weight_delta)


        # estimate params
        # if args.mix_ep == 0:
        if True:
            # agg_params = DescrStatsW(cluster_params_list, cluster_data_ratio_list)
            # agg_gradients = DescrStatsW(cluster_gradients_list, cluster_data_ratio_list)
            agg_params = DescrStatsW(cluster_params_list)
            agg_gradients = DescrStatsW(cluster_gradients_list)

            psi = agg_gradients.var.sum() / total_params  # TODO
            epsilon = np.linalg.norm(agg_gradients.mean) ** 2


            all_params = np.concatenate([agg_params.mean, [psi, epsilon]])
            global_params_list.append(all_params)

            print("Estimated params (global):")
            logger.add_scalar('Estimation/loss', all_params[0], global_step=epoch)
            # logger.add_scalar('Estimation/L', all_params[1], global_step=epoch)
            # logger.add_scalar('Estimation/G', all_params[2], global_step=epoch)
            logger.add_scalar('Estimation/sigma', all_params[1], global_step=epoch)
            logger.add_scalar('Estimation/gamma', all_params[2], global_step=epoch)
            logger.add_scalar('Estimation/G', all_params[3], global_step=epoch)
            logger.add_scalar('Estimation/L', all_params[4], global_step=epoch)
            logger.add_scalar('Estimation/psi', all_params[5], global_step=epoch)
            logger.add_scalar('Estimation/epsilon', all_params[6], global_step=epoch)

            # loss, L, G, sigma, gamma, psi, epsilon = np.mean(global_params_list, axis=0)
            # loss, L, sigma, gamma, G, psi, epsilon = np.mean(global_params_list, axis=0)
            loss, sigma, gamma, G, L, psi, epsilon = np.mean(global_params_list, axis=0)
            tau = args.mix_ep


            # lr = 1 / (8 * L * K)
            # L = min(1 / (8 * lr * K), L)
            init_loss = global_params_list[0][0]
            rho = sigma + 8*K*gamma + 8*K*psi + 8*K*G

            # first_part = init_loss / ( lr* K * (epoch+1))
            # second_part = 5 * lr * K * L**2 * (1 + 1/(K-1))**K * (1 + 2/p) * (tau**2 / p - tau) * rho

            logger.add_scalar('Observation/desirable_lr', 1 / (8 * L * K), global_step=epoch)
            logger.add_scalar('Observation/actual_lr', lr, global_step=epoch)
            logger.add_scalar('Observation/K', K, global_step=epoch)

            # logger.add_scalar('Estimation/first_part', first_part, global_step=epoch)
            # logger.add_scalar('Estimation/second_part', second_part, global_step=epoch)

        # mixing between clusters
        cluster_wegihts_list_mixing = []
        # activate mixing
        if args.mix_ep > 0 and (epoch+1) % args.mix_ep == 0 or \
                args.mix_ep == 0 and mixing_timer <= 0:
            network.activate_mix()
            num_mixing += 1

            for i in range(num_clusters):
                cluster_wegihts_list_mixing.append(mixing_weights(cluster_wegihts_list, topo_matrix[:, i]))
                delta = 0
                for name in cluster_wegihts_list_mixing[i]:
                    delta += (cluster_wegihts_list_mixing[i][name] - cluster_wegihts_list[i][name]).norm() ** 2
                cluster_mix_delta_list[i] = delta.item()

            cluster_wegihts_list = cluster_wegihts_list_mixing
        # deactivate mixing
        else:
            network.deactivate_mix()
            cluster_mix_delta_list = [0 for i in range(num_clusters)]

        # simulate one round in wsn
        traces[routing_topology] = network.simulate_one_round()
        logger.add_scalar('Observation/agg', np.mean(cluster_agg_delta_list), global_step=epoch)
        if network.is_mix:
            logger.add_scalar('Observation/mix', np.mean(cluster_mix_delta_list), global_step=epoch)

        # record energy
        intra_energy_list.append(network.energy_dis['intra-comm'] + network.energy_dis['local-update'])
        inter_energy_list.append(network.energy_dis['inter-comm'])
        logger.add_scalar('Observation/intra', intra_energy_list[-1], global_step=epoch)
        logger.add_scalar('Observation/inter', inter_energy_list[-1], global_step=epoch)

        # tmp
        if num_mixing > 0 :
            inter_energy_per_round = inter_energy_list[-1] / num_mixing
            opt_tau = optimal_mixing_cycles(L, G, sigma, gamma, psi, lr, K, p, init_loss, inter_energy_per_round, budget)
            Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * rho

            # first_part = init_loss * inter_energy_per_round / (lr * K * budget * 1 ** 2)
            # second_part = Gamma * (1 + 2 / p) * (2 * 1 / p - 1)
            # logger.add_scalar('Observation/first_part', first_part, global_step=epoch)
            # logger.add_scalar('Observation/second_part', second_part, global_step=epoch)

            tau_value = 1
            derivative = lr ** 2 * K * L ** 2 * (sigma + psi) * tau_value / p ** 2 - \
            init_loss * inter_energy_per_round / (lr * K * budget * tau_value ** 2)
            logger.add_scalar('Observation/derivative', derivative, global_step=epoch)

            logger.add_scalar('Observation/tau', opt_tau, global_step=epoch)
            logger.add_scalar('Observation/inter_per_round', inter_energy_per_round, global_step=epoch)
        # tmp

        # update mixing cycles
        # mixing_timer = min(mixing_timer, opt_tau)
        if args.mix_ep == 0 and mixing_timer <= 0:

            intra_energy_per_round = intra_energy_list[-1] / len(intra_energy_list)
            inter_energy_per_round = inter_energy_list[-1] / num_mixing

            # calculate optimal mixing cycles
            lr = lr_decay(args.lr, epoch)
            opt_tau = optimal_mixing_cycles(L, G, sigma, gamma, psi, lr, K, p, init_loss, inter_energy_per_round, budget)
            mixing_timer = opt_tau
            print("Optimal mixing cycles: ", opt_tau)

            # Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * rho
            # first_part = init_loss * inter_energy_per_round / (lr * K * budget * opt_tau ** 2)
            # second_part = Gamma * (1 + 2 / p) * (2 * opt_tau / p - 1)
            #
            # logger.add_scalar('Observation/first_part', first_part, global_step=epoch)
            # logger.add_scalar('Observation/second_part', second_part, global_step=epoch)
            # logger.add_scalar('Observation/tau', opt_tau, global_step=epoch)
            # logger.add_scalar('Observation/inter_per_round', inter_energy_per_round, global_step=epoch)

            # adaptive mixing, experimental
            # if np.mean(cluster_agg_delta_list) / intra_energy_per_round < np.mean(cluster_mix_delta_list) / inter_energy_per_round:
            #     dynamic_tau -= 1
            #     dynamic_tau = max(dynamic_tau, 1)
            # else:
            #     dynamic_tau += 1
            # mixing_timer = dynamic_tau
            # logger.add_scalar('Observation/tau', dynamic_tau, global_step=epoch)

        # update mixing matrix
        alive_clusters = network.get_alive_clusters()
        topo_matrix = update_topo(topo_matrix, alive_clusters)
        spectral_norm = np.linalg.norm(topo_matrix - 1 / num_clusters, ord=2) ** 2
        p = 1 - spectral_norm
        logger.add_scalar('Estimation/p', p, global_step=epoch)

        # update global model (used for evaluation)
        global_weights = average_weights(cluster_wegihts_list)


        global_model.load_state_dict(global_weights)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            losses_group = np.array(train_loss_list)[:,-1]
            print(f'Training Loss : {losses_group[losses_group != None].mean()}')
        losses_group = np.array(train_loss_list)[:, -1]
        logger.add_scalar('All/Train/loss', losses_group[losses_group != None].mean(), global_step=epoch)

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
    plot_traces(traces, logger_dir)
    print(traces[routing_topology]['alive_clusters'])
    try:
        plot_time_of_death(network, logger_dir)
    except:
        pass

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
