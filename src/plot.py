from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import json
from collections import OrderedDict
from scipy.signal import savgol_filter

from utils import search_file

config = OrderedDict({
    'budget':  500,
    'dataset': 'fmnist',
    'num_cluster': 5,
    'sim': 0.50,
    'topo': 'ring',
    'mix': [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100],
    'lr': 0.010,
    'frac': 1.0,
})

colors = ['b', 'k', 'y', 'g', 'm', 'c', 'r', 'pick', 'orange']
line_style = ['-', '--', '-.', ':', '-']
markers = ['X', 'P', '*']
fig_size = (8, 6)

log_dir = '../logs/%s/%s' % (config['dataset'], config['topo'])
fig_dir = '../figures'
file_list = os.listdir(log_dir)
file_list = sorted(file_list, reverse=True)  # recent first, file name already includes time information

def get_target_file_list(config, pattern):
    match_items = []
    for k in config:
        if isinstance(config[k], list):
            value_list = config[k]
            match_items += '$'  # placeholder
        else:
            match_items += (config[k],)
    target_file_list = []
    for value in value_list:
        target_file = search_file(pattern, match_items, value, file_list)
        target_file_list.append(target_file)
    return target_file_list, value_list, match_items


def display(data, k='test_acc'):
    if k == 'test_acc':
        return max(data[k])
    elif k == 'test_loss':
        return min(data[k])
    elif k == 'train_loss':
        return min(data[k])
    else:
        exit("Error!")


# # constant mixing cycles
# tau_list = []
# acc_list = []
# for i, target_file in enumerate(target_file_list):
#     file_path = os.path.join(log_dir, target_file, 'convergence.json')
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     tau_list.append(int(value_list[i]))
#     acc_list.append(display(data))
#     # print(max(data['test_acc']))

# dynamic mixing cycles
# target = []
# target_file = search_file(pattern, match_items, 0, file_list)
# file_path = os.path.join(log_dir, target_file, 'convergence.json')
# with open(file_path, 'r') as f:
#     data = json.load(f)
# opt_acc = display(data)
# opt_tau = 4

# path='/data/magnolia/Federated-Learning-PyTorch/logs/2022-03-17 17:33:48_Cluster[5]_Sim[1.00]_Topo[complete]_Mix[1]_lr[0.010]_frac[1.0].pkl'
# ea = event_accumulator.EventAccumulator(path)
# ea.Reload()
# print(ea.scalars.Keys())
# train_loss = [item.value for item in ea.scalars.Items("All/Test/acc")]  # ScalarEvent(wall_time=1647509722.9906468, step=1, value=0.10740000009536743)
# print(train_loss)


# convergence: \tau = 1, 10, 100
pattern = 'Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'
config_copy = config.copy()
config_copy['sim'] = 1.0
config_copy['mix'] = '$'
templates = list(config_copy.values())[1:]

plt.figure(figsize=fig_size)
for i, tau in enumerate([1, 10, 100]):
    config_copy['mix'] = tau
    file_name = search_file(pattern, templates, tau, file_list, exclude='ENERGY')
    file_path = os.path.join(log_dir, file_name, 'convergence.json')
    print(file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    ys = data['test_acc']
    plt.plot(range(len(ys)), ys, linewidth=2,
             linestyle=line_style[i], color=colors[i], label=r'$\tau=%d$' % tau)
plt.ylabel("Test accuracy", fontsize=20)
plt.xlabel("Communication rounds", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
# plt.savefig('%s/convergence_%.2f.pdf' % (fig_dir, config_copy['sim']),
#             format='pdf', dpi=300)
plt.show()

# cluster sim
pattern = '[ENERGY({})]_Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'

plt.figure(figsize=fig_size)
for j, sim in enumerate([0.2]):
    # constant mixing cycles
    tau_list = []
    acc_list = []
    config['sim'] = sim
    target_file_list, value_list, match_items = get_target_file_list(config, pattern)
    # print(target_file_list, value_list, match_items)
    for i, target_file in enumerate(target_file_list):
        file_path = os.path.join(log_dir, target_file, 'convergence.json')
        with open(file_path, 'r') as f:
            data = json.load(f)
        tau_list.append(int(value_list[i]))
        acc_list.append(display(data))

    # smooth data
    # tau_list = tau_list
    # acc_list = savgol_filter(acc_list, 15, 1, mode='nearest')

    plt.plot(tau_list, acc_list, linewidth=2,
             linestyle=line_style[j], color=colors[j], label='Sim.=%.2f' % sim)

    # # # dynamic mixing cycles
    # target = []
    # target_file = search_file(pattern, match_items, 0, file_list)
    # file_path = os.path.join(log_dir, target_file, 'convergence.json')
    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    # opt_acc = display(data)
    # file_path = os.path.join(log_dir, target_file, 'opt_tau_list.json')
    # with open(file_path, 'r') as f:
    #     data = json.load(f)
    # opt_tau = np.mean(data)
    # plt.scatter(opt_tau, opt_acc,
    #             marker=markers[j], alpha=0.5, s=200, color=colors[j])
    # plt.plot([1, 100], [opt_acc, opt_acc], color=colors[j], label='Dynamic')

plt.ylabel("Test accuracy", fontsize=20)
plt.xlabel("Mixing cycles", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xscale('log')
# plt.yscale('log')
plt.legend(fontsize=18)
plt.title(f"Energy budget = {config['budget']} J", fontsize=18)
plt.tight_layout()
# plt.savefig(f'../figures/acc_energy_%d.pdf' % (config['budget']),
#             format='pdf', dpi=300)
plt.show()


# topo


# budget


# convergence

# subplots examples: https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html