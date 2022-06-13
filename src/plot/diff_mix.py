from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import json
from collections import OrderedDict
from scipy.signal import savgol_filter

from utils import search_file

config = OrderedDict({
    'dataset': 'fmnist',
    'num_cluster': 5,
    'sim': 0.80,
    'topo': '$',
    'mix': '$',
    'lr': 0.010,
    'frac': 1.0,
})

colors = ['b', 'k', 'y', 'g', 'm', 'c', 'r', 'pick', 'orange']
line_style = ['-', '--', '-.', ':', '-']
markers = ['X', 'P', '*']
fig_size = (14, 6)
pattern = 'Dataset[{}]_Cluster[{}]_Sim[{:.2f}]_Topo[{}]_Mix[{}]_lr[{:.3f}]_frac[{:.1f}].pkl'
log_base = '../../logs'

fig, ax = plt.subplots(1, 2, figsize=fig_size)
for i, topo in enumerate(['complete']):
    config_instance = config.copy()

    ax = plt.subplot(1, 2, i+1)

    config_instance['topo'] = topo
    log_dir = os.path.join(log_base, '%s/%s' % (config_instance['dataset'], topo))
    file_list = os.listdir(log_dir)
    file_list = sorted(file_list, reverse=True)  # recent first, file name already includes time information
    best_acc_list = []
    for i, tau in enumerate([1, 10, 100]):
        config_instance['mix'] = tau
        templates = list(config_instance.values())
        file_name_list = search_file(pattern, templates, None, file_list, exclude='ENERGY', file_num=5)
        ys = []
        for file_name in file_name_list:
            file_path = os.path.join(log_dir, file_name, 'convergence.json')
            with open(file_path, 'r') as f:
                data = json.load(f)
            ys.append(data['test_acc'])
        y = np.mean(ys, axis=0)
        x = range(len(y))
        best_acc_list.append(max(y))
        plt.plot(x, y, linewidth=2, linestyle=line_style[i], color=colors[i], label=r'$\tau=%d$' % tau)
    print('%8s:\t' % topo + '\t'.join(["%.2f%%" % (acc * 100) for acc in best_acc_list]))

    plt.ylabel("Test accuracy", fontsize=20)
    plt.xlabel("Communication rounds", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=18)
    plt.title(topo, fontsize=20)

plt.tight_layout()

# plt.savefig('%s/convergence_%.2f.pdf' % (fig_dir, config_copy['sim']),
#             format='pdf', dpi=300)
plt.show()
