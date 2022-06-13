from matplotlib import pyplot as plt
from collections import OrderedDict
import json

colors = ['b', 'r', 'k', 'g', 'y', 'c', 'm']
line_style = ['--', '-.', ':', '-']


model_size = 31400
server_num = 36
underlay = 'geantdistance'
file_name = "%d-%s-%d.json.1" % (model_size, underlay, server_num)
with open("./%s" % file_name) as f:
    time_reference = json.load(f)


time = time_reference["100000000.0"]
i = 0
for j, method in enumerate(time):
    print(time[method])
    if len(time[method]) == 1:
        x = range(1, 31)
        color_n_line = colors[j] + line_style[i]
        plt.plot(x, [time[method][0] for _ in x], color_n_line, label=method, linewidth=2)
    else:
        color_n_line = colors[j] + line_style[i]
        plt.plot(range(1, len(time[method])+1), time[method], color_n_line, label=method, linewidth=2)
        i += 1
plt.grid()
plt.legend()
plt.show()
# model = 'LR'
# senario_list = ['geantdistance-40', 'geantdistance-9']
#
# model = 'CNN'
# senario_list = ['geantdistance-9']
#
# nrows = 1; ncols = 1
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols,  figsize=(20,10))
# # fig, ax = plt.subplots(nrows=nrows, ncols=ncols,  figsize=(10, 8))
# legends = time[model]['legend']
#
# for k, senario in enumerate(senario_list):
#     xs = [float(x) / 1e6 for x in time[model][senario].keys()]
#     yy = time[model][senario]
#     ys_list = [[] for _ in range(len(legends))]
#     for x in yy:
#         for i, y in enumerate(yy[x]):
#             # ys_list[i].append(y / yy[x][0])
#             ys_list[i].append(y)
#
#     ax = plt.subplot(nrows, ncols, k + 1)
#     plt.grid()
#     for i, ys in enumerate(ys_list):
#         color_n_line = colors[i] + line_style[i]
#         plt.plot(xs, ys, color_n_line, label=legends[i], linewidth=2)
#         plt.xlabel("Network Capacity (Mbps)", fontsize=20)
#         plt.ylabel("Communication Time (s)", fontsize=20)
#         plt.tick_params(labelsize=20)
#         plt.xscale('log')
#         plt.legend(fontsize=20)
#         plt.title(model+ '-' + senario, fontsize=20)
#         plt.yscale('log')
# plt.show()






