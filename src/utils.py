#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
import os
import torch
import numpy as np
import math

from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, \
    cifar_iid, cifar_noniid, cifar_noniid_cluster
from tsp_christofides import christofides_tsp


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        elif args.dataset == 'fmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def get_dataset_cluster(args, users_groups=None):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        # apply_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.5),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=trans_cifar_train)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=trans_cifar_test)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        elif args.dataset == 'fmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        raise NotImplementedError()
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose equal splits for every user
            user_groups_list, idxs_groups = cifar_noniid_cluster(train_dataset, args.num_users,
                                                       args.num_clusters, args.cluster_similarity, users_groups)

            user_groups_list2, idxs_groups2 = cifar_noniid_cluster(test_dataset, args.num_users, args.num_clusters, 1.0, users_groups)

            split = {
                'train': [user_groups_list, idxs_groups],
                'test':   [user_groups_list2, idxs_groups2],
            }

            # targets = list(np.array(train_dataset.targets)[idxs_groups[0]])  #TODO
            # print([(i, targets.count(i)) for i in range(10)], len(targets))   #TODO

    return train_dataset, test_dataset, split


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(args):
    local_rank = -1 if os.environ.get('LOCAL_RANK') is None else int(os.environ['LOCAL_RANK'])
    if local_rank != -1:
        assert local_rank < torch.cuda.device_count()
        # allocate GPU according to local_rank
        torch.cuda.set_device(local_rank)
        device = 'cuda'
    elif args.gpu is not None:
        # torch.cuda.set_device(0)  #TODO
        torch.cuda.set_device(int(args.gpu))  #TODO
        device = 'cuda'
    elif args.data_parallel is not False:
        # set master GPU
        torch.cuda.set_device(0)
        device = 'cuda'
    else:
        device = 'cpu'

    return device


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def mixing_weights(w, m):
    """
    Returns the mixing ones of the weights by mixing vector.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(len(w)):
            if i == 0:
                w_avg[key] = w[i][key] * m[i]
            else:
                w_avg[key] += w[i][key] * m[i]
    return w_avg


def update_topo(topo, alive_clusters):
    topo = copy.deepcopy(topo)
    for i in range(len(topo)):
        for j in range(len(topo)):
            if i == j:
                topo[i,j] = 1
            if i in alive_clusters and j in alive_clusters and topo[i,j] > 0:
                topo[i,j] = 1
            else:
                topo[i,j] = 0

    new_topo = np.zeros_like(topo)
    matrix_sum = topo.sum(1)
    for i in range(len(topo)):
        for j in range(len(topo)):
            if i != j and topo[i, j] > 0:
                new_topo[i, j] = 1 / max(matrix_sum[i], matrix_sum[j])
        new_topo[i, i] = 1 - new_topo[i].sum()
    return new_topo


def complete(num_clusters):
    matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(num_clusters):
            matrix[i, j] = 1 / num_clusters
    return matrix


def star(num_clusters, centroids=None):
    matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        matrix[i, 0] = 1 / num_clusters
        matrix[i, i] = 1 - 1 / num_clusters
        matrix[0, i] = 1 / num_clusters
    return matrix


def ring(num_clusters, centroids=None):
    assert num_clusters >= 3
    if centroids is None:
        matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            matrix[i,(i+1)%num_clusters] = 1/3
            matrix[i,(i)%num_clusters] = 1/3
            matrix[i,(i-1)%num_clusters] = 1/3
    else:
        assert num_clusters == len(centroids)
        graph = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i+1, num_clusters):
                x1, y1 = centroids[i]
                x2, y2 = centroids[j]
                dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                graph[i, j] = graph[j,i] = dis
        tour = christofides_tsp(graph)
        matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            a, b, c = tour[i], tour[(i+1)%num_clusters], tour[i-1]
            matrix[a, a] = 1/3
            matrix[a, b] = 1/3
            matrix[a, c] = 1/3
    return matrix


def independent(num_clusters):
    matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        matrix[i, i] = 1
    return matrix


def agg_groups(dict_groups, method='mean'):
    # dict_groups = [{'a': Tensor()}, {'a': Tensor()}, ...]
    mean = {}
    for i in range(len(dict_groups)):
        for k in dict_groups[i]:
            if i == 0:
                mean[k] = copy.deepcopy(dict_groups[i][k]) / len(dict_groups)
            else:
                mean[k] += dict_groups[i][k] / len(dict_groups)

    var = {}
    for i in range(len(dict_groups)):
        for k in dict_groups[i]:
            if i == 0:
                var[k] = np.linalg.norm(copy.deepcopy(dict_groups[i][k]) - mean[k])**2 / len(dict_groups)
            else:
                var[k] += np.linalg.norm(dict_groups[i][k] - mean[k])**2 / len(dict_groups)

    return mean, var


def optimal_mixing_cycles(L, G, sigma, gamma, psi, lr, K, p, init_loss, inter_cost, budget):
    if L == 0 :
        return 1
    # rho = sigma + 8 * K * gamma + 8 * K * psi + 8 * K * G

    # Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * rho
    # Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * rho
    # def derivative(tau):
    #     return Gamma * (1 + 2/p) * (2*tau/p - 1) - init_loss * inter_cost / ( lr * K * budget* tau**2)

    def derivative(tau):
        return 12 * lr ** 2 * K * L ** 2 * (sigma + 6 * K * psi) * (1 + 2 / p) * (2 * tau / p - 1) - \
               init_loss * inter_cost / (lr * K * budget * tau ** 2)
        # return lr ** 2 * K * L ** 2 * (sigma + K * psi) * tau / p **2 - \
        #        init_loss * inter_cost / ( lr * K * budget* tau**2)

    rs = fsolve(derivative, 1)
    return max(round(rs[0]), 1)


def lr_decay(lr, global_round, method=None, steps=50):
    if method is None:
        return lr
    elif method == 'sqrt':
        return lr / math.sqrt(global_round // steps + 1)
    else:
        raise NotImplementedError()

    # return lr * 0.8 ** (round//50)  # make sense

    # decay = 0.5
    # lr = lr / np.sqrt(1 + decay * round)
    # lr = lr / (1 + decay * round)


def plot_label_dis(x, y):
    fig = plt.figure()
    plt.bar(x, y)
    return fig


def plot_user_dis(train_dataset, user_groups_list, num_classes, num_users, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    targets_groups = []
    for user_groups in user_groups_list:
        for u in user_groups:
            targets, counts = np.unique(np.array(train_dataset.targets)[user_groups[u]], return_counts=True)
            idx_counts = np.argsort(counts)[::-1]  # sort label according to its frequency
            targets = targets[idx_counts][:2]  # obtain 2 labels at most
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
            Z[i, j] = c
    z = Z.ravel() / num_users
    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1
    ax.bar3d(x, y, bottom, width, depth, z, shade=True)
    ax.view_init(elev=20., azim=-45)
    ax.set_xlabel("Label", fontsize=18, labelpad=10)
    ax.set_ylabel("Label", fontsize=18, labelpad=10)
    ax.set_zlabel("User ratio", fontsize=18, labelpad=10)
    ax.xaxis.set_ticks([1, 3, 5, 7, 9])
    ax.yaxis.set_ticks([1, 3, 5, 7, 9])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'user_dis.pdf'),  format='pdf', dpi=300,
                    bbox_inches='tight',pad_inches = -0.03)
    # surf = ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, cmap='viridis', edgecolor='none')
    # surf = ax.plot_surface(X,Y,Z, linewidth=0)
    # ax.plot_wireframe(X,Y,Z, rstride=10, cstride=0)
    # ax.zaxis.set_major_locator(LinearLocator(6))
    return fig


# def search_file(pattern, match_items, value, file_list, placeholder='$', exclude=None):
#     # print(pattern, match_items, value)
#     # put together search content
#     target = []
#     for item in match_items:
#         if item == placeholder:
#             target.append(value)
#         else:
#             target.append(item)
#     target_name = pattern.format(*target)
#     is_find = False
#     for name in file_list:
#         if target_name in name:
#             if exclude is not None and exclude in name:
#                 continue
#             target_file = name
#             is_find = True
#             break
#     if not is_find:
#         target_file = None
#
#     return target_file

def search_file(pattern, match_items, value, file_list, placeholder='$', exclude=None, file_num=1):
    # print(pattern, match_items, value)
    # put together search content
    target = []
    for item in match_items:
        if item == placeholder:
            target.append(value)
        else:
            target.append(item)
    target_name = pattern.format(*target)
    target_file_list = []
    for name in file_list:
        if target_name in name:
            if exclude is not None and exclude in name:
                continue
            target_file_list.append(name)
            if len(target_file_list) == file_num:
                break
    assert len(target_file_list) == file_num
    return target_file_list

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def exp_details_cluster(args):
    print('\nFederated parameters:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    if args.local_iter > 0:
        print(f'    Local Iterations   : {args.local_iter}')
    else:
        print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Mix cycles     : {args.mix_ep}')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}\n')

    print('Network details:')
    print(f'    Users       : {args.num_users}')
    print(f'    Clusters    : {args.num_clusters}')
    print(f'    Topology    : {args.topo}')
    print(f'    Cluster Similarity : {args.cluster_similarity}\n')

    return


if __name__ == '__main__':
    # L, G, sigma, gamma, psi, p = 10, 100, 0.5, 5, 2, 1
    # lr = 0.0001
    # K = 10
    # # lr=1/(8*L*K)
    # init_loss=2
    # # print("L=%2f, G=%2f, sigma=%2f, gamma=%2f, psi=%2f, p=%2f" % (L, G, sigma, gamma, psi, p))
    #
    # Gamma = 5 * lr ** 2 * K * L ** 2 * (1 + 1 / (K - 1)) ** K * (sigma + 8 * K * gamma + 8 * K * psi + 8 * K * G)
    #
    # cost = 1
    # budget = 500
    # tau=1
    # print(Gamma* (1 + 2 / p) * (2 * tau / p - 1))
    # print(init_loss / (lr * K * tau ** 2) * cost / budget)
    #
    # exit(1)
    #
    # def derivative(tau):
    #     # print(Gamma * (1+2/p)*(2*tau/p-1), init_loss/(lr*K*tau**2) * cost/budget)
    #     return Gamma * (1 + 2 / p) * (2 * tau / p - 1) - init_loss / (lr * K * tau ** 2) * cost / budget
    # print(derivative(2))

    # from matplotlib import pyplot as plt
    # import numpy as np
    #
    # x=[]
    # y=[]
    # for tau in [1,2,4]:
    #     for i in range(1000):
    #         x.append(i)
    #         y.append(np.log(i)+i*0.001*(tau**2-tau))
    #
    #     plt.plot(x,y)
    #     plt.xlim([10,None])
    #     plt.show()

    import numpy as np
    tau = [2, 4, 8]
    epsilon = [49.2895, 54.9271, 55.8868]
    p=1
    def f(eps1, tau1, eps2, tau2):
        return(eps1-eps2)/((1+2/p)*(tau1**2/p - tau1) - (1+2/p)*(tau2**2/p - tau2))

    def f3(eps1, tau1, eps2, tau2):
        return(eps1-eps2)/((1+2/p)*(np.sqrt(tau1)) - (1+2/p)*(np.sqrt(tau2)))

    print(f(epsilon[0], tau[0], epsilon[1], tau[1]))
    print(f(epsilon[0], tau[0], epsilon[2], tau[2]))
    print(f(epsilon[1], tau[1], epsilon[2], tau[2]))