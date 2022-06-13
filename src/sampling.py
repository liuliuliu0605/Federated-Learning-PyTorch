#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid_cluster(dataset, num_users, num_clusters=1, cluster_similarity=0., users_groups=None):
    """
    Group and sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param num_clusters:
    :param cluster_similarity: 0~1
    :return:
    """
    # if users_groups is not None:
    #     assert num_users == np.concatenate(users_groups).size
    #     assert num_clusters == len(users_groups)
    #     # idxs_users_groups = np.vstack((users_groups, range(num_clusters)))
    #     # idxs_users_groups.sort(key=lambda x: len(x))
    #     users_groups.sort(key=lambda  x:len(x))

    labels = np.array(dataset.targets)
    num_labels = len(labels)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if users_groups is not None:
        # group accroding to users_groups
        assert num_users == np.concatenate(users_groups).size
        assert num_clusters == len(users_groups)
    else:
        # group equally
        num_users_per_cluster = num_users// num_clusters
        users_groups = [range(i*num_users_per_cluster, (i+1)*num_users_per_cluster) for i in range(num_clusters)]

    # group labels (cluster_similarity = 0)
    data_split_idxs = [0] + np.cumsum([int(len(users) / num_users * num_labels) for users in users_groups]).tolist()
    idxs_groups_initial = [idxs[data_split_idxs[i]:data_split_idxs[i+1]] for i in range(num_clusters)]

    percent_groups = np.array(data_split_idxs) / num_labels

    # mix groups according to cluster_similarity
    assert cluster_similarity >= 0. and cluster_similarity <= 1.
    idxs_groups = [[] for _ in range(num_clusters)]
    for c, idxs in enumerate(idxs_groups_initial):
        mix_size = int(len(idxs_groups_initial[c]) * cluster_similarity)
        mix_size_groups = mix_size * percent_groups

        idxs_mixed = np.random.choice(idxs, mix_size, replace=False).tolist()
        idxs_left = list(set(idxs) - set(idxs_mixed))
        for i in range(num_clusters):
            idxs_groups[i] += idxs_mixed[int(mix_size_groups[i]):int(mix_size_groups[i+1])]  # may remain ??
        idxs_groups[c] += idxs_left
    # if cluster_similarity >= 1:
    #     num_per_class = len(labels) // num_classes
    #     num_per_class_per_cluster =  num_per_class // num_clusters
    #     print(num_per_class, num_per_class_per_cluster)
    #     for c in range(num_clusters):
    #         for j in range(num_classes):
    #             idxs_groups[c] += idxs[j*num_per_class+c*num_per_class_per_cluster:
    #                                    j*num_per_class+(c+1)*num_per_class_per_cluster].tolist()

    # divide and assign for each cluster, each user randomly choose num_shards_per_user ( default 2) shards
    dict_users_list = []
    left_idxs = []
    num_shards = 200
    num_imgs = num_labels // num_shards
    num_shards_per_user = 2

    for c, idxs in enumerate(idxs_groups):
        my_shards = int(num_shards * len(users_groups[c]) / num_users)
        idx_shard = [i for i in range(my_shards)]
        if users_groups is not None:
            dict_users = {u: np.array([], dtype=int) for u in users_groups[c]}
        else:
            cluster_size = num_users // num_clusters
            dict_users = {u: np.array([], dtype=int) for u in range(c * cluster_size, (c + 1) * cluster_size)}

        # sort labels
        idxs_labels = np.vstack((idxs, labels[idxs]))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        for i, u in enumerate(dict_users.keys()):  # num_users * num_shards_per_user <= num_shards
            # if len(idx_shard) < 2:
            #     for rand in idx_shard:
            #         dict_users[u] = np.concatenate(
            #             (dict_users[u], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            #     for _ in range(2-len(idx_shard)):
            #         dict_users[u] = np.concatenate(
            #             (dict_users[u], left_idxs[0:num_imgs]), axis=0)
            #         left_idxs = left_idxs[num_imgs:]
            # else:
            rand_set = set(np.random.choice(idx_shard, num_shards_per_user, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[u] = np.concatenate(
                    (dict_users[u], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            np.random.shuffle(dict_users[u])

        for rand in idx_shard:
            left_idxs.extend(idxs[rand * num_imgs:(rand + 1) * num_imgs])

        dict_users_list.append(dict_users)
        # dict_users_list[idxs_users_groups[c][0]] = dict_users

    if users_groups is not None:
        idxs_groups = [[] for _ in range(num_clusters)]
        for c, dict_users in enumerate(dict_users_list):
            for user in dict_users:
                idxs_groups[c].extend(dict_users[user])

    return dict_users_list, idxs_groups


if __name__ == '__main__':
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,),
    #                                                         (0.3081,))
    #                                ]))
    # num = 100
    # d = mnist_noniid(dataset_train, num)

    dataset_train = datasets.CIFAR10('../data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))
    num_users, num_clusters, cluster_similarity = 100, 5, 1
    users_groups = [[i for i in range(20)], [i for i in range(20, 40)], [i for i in range(40, 60)],
                    [i for i in range(60, 80)], [i for i in range(80, 100)]]
    # users_groups = None
    x = users_groups[0].pop(-1)
    users_groups[-1].append(x)

    dict_users_list, idxs_groups = cifar_noniid_cluster(dataset_train, num_users, num_clusters, cluster_similarity,
                                                        users_groups)

    import pandas as pd
    for c in range(num_clusters):
        print("\n Cluster %d:" % c)
        print("labels:", set(np.array(dataset_train.targets)[idxs_groups[c]]),
              len(np.array(dataset_train.targets)[idxs_groups[c]]))
        targets = list(np.array(dataset_train.targets)[idxs_groups[c]])
        print("label distribution:", [(i, targets.count(i)) for i in range(10)], len(targets))
        my_users = [user for user in dict_users_list[c]]
        print("%d users:" % len(my_users), my_users)
        user_labels_groups = [list(pd.value_counts((np.array(dataset_train.targets)[dict_users_list[c][u]])).items())
                              for u in dict_users_list[c]]
        print(user_labels_groups)

    from torch.utils.data import DataLoader

    trainloader = DataLoader(dataset_train, batch_size=10, shuffle=True)


