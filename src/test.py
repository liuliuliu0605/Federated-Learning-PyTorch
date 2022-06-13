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

from models import *
from torchstat import stat
import numpy as np
from functools import reduce


def power_two(num):
    return num&num-1 == 0


def T_r(p, n, alpha, w):
    return (p-1) * alpha + (p-1)/p * (n/w)


def T_c(p, n , alpha, w):
    return np.log2(p) * alpha + (p-1)/p * (n/w)


def T_rc(p, n, alpha, w):
    if power_two(int(p)):
        return T_c(p, n, alpha, w)
    else:
        return T_r(p, n, alpha, w)


def T_one_lvl(p_list, n , alpha, w_list):
    return 2*T_r(reduce(lambda x, y: x*y, p_list), n, alpha, min(w_list))


def T_two_lvl(p_list, n , alpha, w_list):
    return 2*T_c(p_list[0], n, alpha, w_list[0]) + \
           2*T_r(p_list[0], n, alpha, w_list[0]) + \
           2*T_rc(reduce(lambda x, y: x*y, p_list)/p_list[0], n, alpha, min(w_list))


def T_blc(p_list, n, alpha, w_list):
    rs = 0
    for i in range(len(w_list)):
        n_i = n if i == 0 else n/reduce(lambda x, y: x*y, p_list[:i])
        w_is = [w_list[0]]
        if i > 0:
            w_is.extend([w_list[i] / reduce(lambda x, y: x*y, p_list[:j+1]) for j in range(0, i)])
        w_i = min(w_is)
        rs += T_rc(p_list[i], n_i, alpha, w_i)
        # print(p_list[i], n_i, alpha, w_i)
    return 2*rs


def T_fedpma(p_list, n, alpha, w_list):
    return alpha + n / min(w_list)


p = 10
n = 100
alpha = 0.1
p_list = [3, 2, 2]
# w_list = [256, 100, 50]
w_list = [32*8, 32*8/3, 32*8/6]


print('one level:', T_one_lvl(p_list, n, alpha, w_list))
print('two level:', T_two_lvl(p_list, n, alpha, w_list))
print('blue connect:', T_blc(p_list, n, alpha, w_list))
print('fedpma:', T_fedpma(p_list, n, alpha, w_list))
# a=10000000
# b=1
# E = 100
# e1 = 1
# e2 = 1
#
# def f(tau, T_tau):
#     # print(a / T_tau / tau, b * tau**2)
#     return -(a / T_tau / tau + b * tau**2)
#
# def g(tau):
#     print(tau, E / (e1 + tau * e2) )
#     return -(a * (e1 + tau * e2) / (E * tau) + b * tau **2)
#
#
# x = []
# y = []
# for tau in range(1, 100):
#     # min_f = 999999
#     # opt_T_tau = None
#     max_T_tau = int(E / (e1 + tau * e2))
#     # print(max_T_tau, E / (e1 + tau * e2))
#     min_f = f(tau, max_T_tau)
#     # print(max_T_tau, tau)
#     # for T_tau in range(1, max_T_tau + 1):
#     #     v = f(tau, T_tau)
#     #     if v < min_f:
#     #         min_f = v
#     #         opt_T_tau = T_tau
#     x.append(tau)
#     y.append(min_f)
# plt.plot(x, y)
#
# xx=[]
# yy=[]
# for tau in range(1, 500):
#     xx.append(tau)
#     yy.append(g(tau))
# plt.plot(xx, yy)
# plt.xscale('log')
# plt.show()



# len_in = 28 * 28
# global_model = LR(dim_in=len_in, dim_out=10)
# print(global_model)
# stat(global_model, (1, 28, 28))


# if __name__ == '__main__':
#     L = 10
#     G = 0
#     sigma = 0
#     gamma = 0
#     psi = 0
#     lr=0.01
#     K=10
#     p=1
#     init_loss=2
#     inter_cost = 1
#     budget =100
#     rs = optimal_mixing_cycles(L, G, sigma, gamma, psi, lr, K, p, init_loss, inter_cost, budget)
#     print(rs)


# fake = False
#
# if __name__ == '__main__':
#     start_time = time.time()
#
#     args = args_parser()
#     print(args)
#     init_seed(args.seed)
#     exp_details_cluster(args)
#     device = get_device(args)
#     print(args.lr)

    # # load dataset and user groups list
    # train_dataset, test_dataset, user_groups_list = get_dataset_cluster(args)
    #
    # # BUILD MODEL
    # if args.model == 'cnn':
    #     # Convolutional neural netork
    #     if args.dataset == 'mnist':
    #         global_model = CNNMnist(args=args)
    #     elif args.dataset == 'fmnist':
    #         global_model = CNNFashion_Mnist(args=args)
    #     elif args.dataset == 'cifar':
    #         global_model = CNNCifar(args=args)
    #
    # elif args.model == 'mlp':
    #     # Multi-layer preceptron
    #     img_size = train_dataset[0][0].shape
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #         global_model = MLP(dim_in=len_in, dim_hidden=64,
    #                            dim_out=args.num_classes)
    # else:
    #     exit('Error: unrecognized model')

#
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn.functional as F
# torch.manual_seed(10)
#
#
# # ============================ step 1/5 生成数据 ============================
# sample_nums = 100
# mean_value = 1.7
# bias = 1
# n_data = torch.ones(sample_nums, 4)
# x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
# y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1)
# x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
# y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1)
# train_x = torch.cat((x0, x1), 0)
# train_y = torch.cat((y0, y1), 0)
#
#
# # ============================ step 2/5 选择模型 ============================
# class LR(nn.Module):
#     def __init__(self):
#         super(LR, self).__init__()
#         self.features = nn.Linear(2, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.sigmoid(x)
#         return x
#
# class CNNCifar(nn.Module):
#     def __init__(self):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 1)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
#
# lr_net = LR()   # 实例化逻辑回归模型
# lr_net = CNNCifar()   # 实例化逻辑回归模型
#
#
# # ============================ step 3/5 选择损失函数 ============================
# loss_fn = nn.BCELoss()
#
# # ============================ step 4/5 选择优化器   ============================
# lr = 0.01  # 学习率
# optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)
#
# # ============================ step 5/5 模型训练 ============================
# for iteration in range(100):
#
#     # 前向传播
#     y_pred = lr_net(train_x)
#
#     # 计算 loss
#     loss = loss_fn(y_pred.squeeze(), train_y)
#
#     # 反向传播
#     loss.backward()
#     G = 0
#     for name, param in lr_net.named_parameters():
#         G += param.grad.norm() ** 2
#     print(G)
#
#
#     # 更新参数
#     optimizer.step()
#
#     # 清空梯度
#     optimizer.zero_grad()
#
#     # 绘图
#     if iteration % 20 == 0:
#
#         mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
#         correct = (mask == train_y).sum()  # 计算正确预测的样本个数
#         acc = correct.item() / train_y.size(0)  # 计算分类准确率
#
#         plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
#         plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')
#
#         w0, w1 = lr_net.features.weight[0]
#         w0, w1 = float(w0.item()), float(w1.item())
#         plot_b = float(lr_net.features.bias[0].item())
#         plot_x = np.arange(-6, 6, 0.1)
#         plot_y = (-w0 * plot_x - plot_b) / w1
#
#         # init_weights_grads = {name: (param.clone().detach(), param.grad.clone().detach())
#         #                       for name, param in lr_net.named_parameters()}
#
#         plt.xlim(-5, 7)
#         plt.ylim(-7, 7)
#         plt.plot(plot_x, plot_y)
#
#         plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
#         plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
#         plt.legend()
#
#         plt.show()
#         plt.pause(0.5)
#
#         if acc > 0.99:
#             break


# from sklearn.datasets import make_blobs
# from matplotlib import  pyplot as plt
#
# centers = [[70, 200], [50, 100], [120, 50], [220, 80], [180, 210]]
# X, y = make_blobs(n_samples=100, centers=centers, n_features=2, random_state=0,
#                   cluster_std=15)
# plt.scatter(X[:,0], X[:,1])
# plt.show()
