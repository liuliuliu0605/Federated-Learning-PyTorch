#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import os
import copy
import json
import datetime, time
import numpy as np
import operator
from tensorboardX import SummaryWriter
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm
from sklearn.datasets import make_blobs

import config
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LR
from utils import get_dataset_cluster, average_weights, mixing_weights, exp_details_cluster, get_device, init_seed,\
    complete, star, ring, independent, lr_decay, \
    plot_label_dis, plot_user_dis, optimal_mixing_cycles, update_topo

from simulator.wsn.network import Network
from simulator.wsn.utils import *
from simulator.wsn.fcm import *
from statsmodels.stats.weightstats import DescrStatsW


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    print(args)
    init_seed(args.seed)
    exp_details_cluster(args)
    device = get_device(args)
    fake = args.fake
    budget = config.energy_budget

    # Define paths
    tag = 'Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'. \
        format(args.dataset, args.num_clusters, args.cluster_similarity, args.topo, args.mix_ep, args.lr, args.frac)
    cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    file_name = '%s_[ENERGY(%d)]_%s' % (cur_datetime, budget, tag) if not fake else \
        '%s_[ENERGY(%d)]_%s.fake' % (cur_datetime, budget, tag)
    log_location = args.log_location
    logger_dir = os.path.join(log_location, file_name)
    logger = SummaryWriter(logger_dir)

    if fake:
        file_list = os.listdir(log_location)
        file_list = sorted(file_list, reverse=True)  # recent first, file name already includes time information
        is_find = False
        for name in file_list:
            if tag in name and 'fake' not in name:  # fake result can not be used for fake training
                fake_path = os.path.join(log_location, name)
                is_find = True
                break
        if is_find:
            print("fake path:", fake_path)
            with open(os.path.join(fake_path, 'convergence.json'), 'r') as f:
                fake_rs = json.load(f)

    # Build topo
    num_clusters = args.num_clusters
    topo_matrix = None
    if args.topo == 'complete':
        topo_matrix = complete(num_clusters)
    elif args.topo == 'star':
        topo_matrix = star(num_clusters, centroids=config.centroids)
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

    # Setup wsn networks and routing protocol
    # We use make_blobs to initialize nodes with locations by specifying centroids
    # Nodes can also be initialized by Network itself with init_nodes=None, and the nodes are uniformly
    # distributed in a area AREA_WIDTH * AREA_LENGTH defined in config.py
    traces = {}
    init_nodes, y = make_blobs(n_samples=args.num_users, centers=config.centroids, n_features=2,
                               random_state=args.seed, cluster_std=15)
    network = Network(init_nodes=init_nodes)
    network.set_topo(topo_matrix)
    routing_protocol_class = eval(config.routing_topology)
    network.init_routing_protocol(routing_protocol_class())
    users_groups = network.get_cluster_members()
    print("Clusters: ", users_groups)
    plot_clusters(network, logger_dir)

    # Load dataset and split users into clusters.
    # If config.users_groups is None, the number of users in each cluster is equal.
    # Otherwise, the size is the same with config.users_groups.
    train_dataset, test_dataset, split = get_dataset_cluster(args, users_groups=users_groups)
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
    fig2 = plot_user_dis(train_dataset, user_groups_list, num_classes, args.num_users, save_path=logger_dir)  # all user distribution
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

    # Copy weights to each cluster.
    global_weights = global_model.state_dict()
    cluster_wegihts_list = [copy.deepcopy(global_weights) for i in range(num_clusters)]

    # Observations in each cluster.
    cluster_params_list = [None for _ in range(num_clusters)]
    cluster_gradients_list = [None for _ in range(num_clusters)]
    cluster_data_ratio_list = [None for _ in range(num_clusters)]
    cluster_agg_delta_list = [0 for i in range(num_clusters)]
    cluster_mix_delta_list = [0 for i in range(num_clusters)]


    # Training and testing results
    train_loss_list, train_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]
    test_loss_list, test_accuracy_list = [[] for _ in range(num_clusters)], [[] for _ in range(num_clusters)]

    global_params_list = []
    opt_tau, mixing_timer, num_mixing = 1, 1, 0
    intra_energy_list, inter_energy_list = [], []
    opt_tau_list = []
    total_energy = network.get_remaining_energy()

    print_every = 2
    top_k = 0.2
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        print("Remain energy:", network.get_remaining_energy())
        logger.add_scalar('Observation/energy_consumption', total_energy - network.get_remaining_energy(), global_step=epoch)
        if network.get_remaining_energy() <= total_energy - budget:
            print("\n Terminated for using up energy budget (%d J)!" % budget)
            break

        if fake and args.mix_ep > 0:
            assert operator.eq(users_groups, config.users_groups)
            train_loss = fake_rs['train_loss'][epoch]
            test_loss = fake_rs['test_loss'][epoch]
            test_accuracy = fake_rs['test_acc'][epoch]
            logger.add_scalar('All/Test/acc', test_accuracy, global_step=epoch)
            logger.add_scalar('All/Test/loss', test_loss, global_step=epoch)
            logger.add_scalar('All/Train/loss', train_loss, global_step=epoch)
            final_epoch = epoch

            if args.mix_ep > 0 and (epoch + 1) % args.mix_ep == 0:
                network.activate_mix()
            else:
                network.deactivate_mix()
            traces[config.routing_topology] = network.simulate_one_round()

            # record energy
            intra_energy_list.append(network.energy_dis['intra-comm'] + network.energy_dis['local-update'])
            inter_energy_list.append(network.energy_dis['inter-comm'])
            inter_intra_ratio = np.mean(inter_energy_list) / np.mean(intra_energy_list)
            # print("******", np.mean(inter_energy_list), np.mean(intra_energy_list))
            logger.add_scalar('Observation/intra', intra_energy_list[-1], global_step=epoch)
            logger.add_scalar('Observation/inter', inter_energy_list[-1], global_step=epoch)
            logger.add_scalar('Observation/inter_intra_ratio', inter_intra_ratio, global_step=epoch)

            continue

        lr = lr_decay(args.lr, epoch)
        logger.add_scalar('Observation/lr', lr, global_step=epoch)
        K = len(train_dataset) / args.num_users / args.local_bs  # 100 is the number of users

        # Used in dynamic mixing when decreasing to 0
        mixing_timer -= 1

        # Aggregation in each cluster
        total_num_participation = 0
        for c, user_groups in enumerate(user_groups_list):
            local_weights, local_losses = [], []
            global_model.load_state_dict(cluster_wegihts_list[c])

            # test
            test_acc, test_loss = test_inference(args, global_model, test_dataset, test_idxs_list[c])
            test_accuracy_list[c].append(test_acc)
            test_loss_list[c].append(test_loss)

            # train
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
                print("Cluster %d is dead !" % c)

            # select participating users, which should be alive
            m = max(int(args.frac * total), 1)
            m = min(m, alive)
            idxs_users = np.random.choice(my_users, m, replace=False)

            # train models and estimate params
            params_groups = []
            init_grad_groups = []
            local_weight_delta = 0

            for idx in idxs_users:
                # initialize local model in each participating user
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)

                # local updates and estimate params
                rs_params = {}
                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,
                                                     rt_params=rs_params, num_params=total_params)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

                params_groups.append(rs_params['estmt_params'])

                # # approximate
                # init_grad_vector = []
                # for name in cluster_wegihts_list[c]:
                #     approximate_grad = (cluster_wegihts_list[c][name] - w[name]) / lr
                #     init_grad_vector.extend(approximate_grad.cpu().numpy().reshape(-1).tolist())
                # init_grad_groups.append(init_grad_vector)

                # only get top-k dimension of init gradient
                top_idxs = np.argsort(np.abs(rs_params['init_grad_vector']))[0:int(total_params*(1-top_k))]
                rs_params['init_grad_vector'][top_idxs] = 0

                init_grad_groups.append(rs_params['init_grad_vector'])

                local_weight_delta += rs_params['weight_delta']
            local_weight_delta /= len(idxs_users)

            # aggregate models and params in a cluster
            agg_weights = average_weights(local_weights)
            cluster_gradients_list[c] = np.mean(init_grad_groups, axis=0)
            cluster_data_ratio_list[c] = len(idxs_users)
            partial_params = np.mean(params_groups, axis=0)

            gamma = np.var(init_grad_groups, axis=0).sum() / total_params
            G = np.linalg.norm(cluster_gradients_list[c]) ** 2 / total_params

            agg_weight_delta, local_grad_delta, index = 0, 0, 0
            for name, param in global_model.named_parameters():
                agg_weight_delta += (agg_weights[name] - cluster_wegihts_list[c][name]).norm() ** 2
                local_grad_delta += np.linalg.norm(
                    (cluster_wegihts_list[c][name] - agg_weights[name]).cpu().numpy().reshape(-1) / lr
                    - K * cluster_gradients_list[c][index:index + agg_weights[name].numel()]) ** 2
                index += param.numel()
            L = np.sqrt(local_grad_delta / local_weight_delta / K)

            cluster_params_list[c] = np.concatenate([partial_params, [gamma, G, L]])
            cluster_agg_delta_list[c] = agg_weight_delta.item()
            cluster_wegihts_list[c] = agg_weights
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss_list[c].append(loss_avg)
            logger.add_scalar('Cluster/%d/Train/loss' % c, loss_avg, global_step=epoch)
            total_num_participation += len(idxs_users)

        logger.add_scalar('Observation/agg', np.mean(cluster_agg_delta_list), global_step=epoch)

        # Collect params from all clusters and obtain global params
        for i in range(num_clusters):
            # only get top-k dimension of init gradient
            top_idxs = np.argsort(np.abs(cluster_gradients_list[i]))[0:int(total_params * (1 - top_k))]
            cluster_gradients_list[i][top_idxs] = 0
        agg_params = DescrStatsW(cluster_params_list, weights=cluster_data_ratio_list)
        agg_gradients = DescrStatsW(cluster_gradients_list, weights=cluster_data_ratio_list)

        psi = agg_gradients.var.sum() / total_params  # TODO, solved ??
        epsilon = np.linalg.norm(agg_gradients.mean) ** 2

        global_params = np.concatenate([agg_params.mean, [psi, epsilon]])
        global_params_list.append(global_params)

        print("Global estimated params:")
        for i, name in enumerate(['loss', 'sigma', 'gamma', 'G', 'L', 'psi', 'epsilon']):
            logger.add_scalar('Estimation/%s' % name, global_params[i], global_step=epoch)

        # Mix between clusters
        cluster_wegihts_list_mixing = []
        if args.mix_ep > 0 and (epoch + 1) % args.mix_ep == 0 or \
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
            logger.add_scalar('Observation/mix', np.mean(cluster_mix_delta_list), global_step=epoch)
        else:
            network.deactivate_mix()
            cluster_mix_delta_list = [0 for i in range(num_clusters)]

        global_weights = average_weights(cluster_wegihts_list)
        delta = 0
        for i in range(num_clusters):
            for name in global_weights:
                delta += (global_weights[name] - cluster_wegihts_list[i][name]).norm() ** 2
        delta /= num_clusters
        logger.add_scalar('Observation/model_consensus', delta, global_step=epoch)

        # simulate one round in wsn
        traces[config.routing_topology] = network.simulate_one_round()
        if network.is_mix:
            logger.add_scalar('Observation/mix', np.mean(cluster_mix_delta_list), global_step=epoch)

        # record energy
        intra_energy_list.append(network.energy_dis['intra-comm'] + network.energy_dis['local-update'])
        inter_energy_list.append(network.energy_dis['inter-comm'])
        inter_intra_ratio = np.mean(inter_energy_list) / np.mean(intra_energy_list)
        logger.add_scalar('Observation/intra', intra_energy_list[-1], global_step=epoch)
        logger.add_scalar('Observation/inter', inter_energy_list[-1], global_step=epoch)
        logger.add_scalar('Observation/inter_intra_ratio', inter_intra_ratio, global_step=epoch)

        # mixing_timer = min(mixing_timer, opt_tau)
        if args.mix_ep == 0 and mixing_timer <= 0:

            intra_energy_per_round = intra_energy_list[-1] / len(intra_energy_list)
            inter_energy_per_round = inter_energy_list[-1] / num_mixing

            # calculate optimal mixing cycles
            loss, sigma, gamma, G, L, psi, epsilon = np.mean(global_params_list, axis=0)
            init_loss = global_params_list[0][0]
            opt_tau = optimal_mixing_cycles(L, G, sigma, gamma, psi, lr, K, p, init_loss, inter_energy_per_round, budget)
            mixing_timer = opt_tau
            opt_tau_list.append(opt_tau)
            logger.add_scalar('Observation/tau', opt_tau, global_step=epoch)

        # update mixing matrix
        alive_clusters = network.get_alive_clusters()
        topo_matrix = update_topo(topo_matrix, alive_clusters)
        spectral_norm = np.linalg.norm(topo_matrix - 1 / num_clusters, ord=2) ** 2
        p = 1 - spectral_norm
        logger.add_scalar('Estimation/p', p, global_step=epoch)

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

    if not fake:
        with open(os.path.join(logger_dir, 'convergence.json'), 'w') as f:
            rs = {
                'train_loss': DescrStatsW(train_loss_list, cluster_data_ratio_list).mean.tolist(),
                'test_loss': DescrStatsW(test_loss_list, cluster_data_ratio_list).mean.tolist(),
                'test_acc': DescrStatsW(test_accuracy_list, cluster_data_ratio_list).mean.tolist()
            }
            f.write(json.dumps(rs, indent=4))
            # json.dump(dict, f, indent=4)
    else:
        with open(os.path.join(logger_dir, 'convergence.json'), 'w') as f:
            rs = {
                'train_loss': fake_rs['train_loss'][:final_epoch],
                'test_loss': fake_rs['test_loss'][:final_epoch],
                'test_acc': fake_rs['test_acc'][:final_epoch]
            }
            f.write(json.dumps(rs, indent=4))

    with open(os.path.join(logger_dir, 'user_groups_list.json'), 'w') as f:
        f.write(json.dumps(network.get_cluster_members()))

    with open(os.path.join(logger_dir, 'energy_consumption.json'), 'w') as f:
        rs = {
            'intra': inter_energy_list,
            'inter': inter_energy_list,
        }
        f.write(json.dumps(rs))

    if args.mix_ep == 0:
        with open(os.path.join(logger_dir, 'opt_tau_list.json'), 'w') as f:
            f.write(json.dumps(opt_tau_list))

    # plot cluster information
    # plot_clusters(network, logger_dir)
    # plot_time_of_death(network, logger_dir)
    # plot_traces(traces, logger_dir)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))