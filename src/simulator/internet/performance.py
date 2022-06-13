import numpy as np
from src.simulator.internet.network import Network
import warnings
from collections import OrderedDict
import json
import ns.core
import os
import time

warnings.filterwarnings("ignore")

seed = 0

# ###########################################setting###################################################
num_of_iterations = 1
# baselines = ['hier_favg', 'complete-1', 'ring-based_allreduce',
#              'neighbour-1', 'neighbour-25',
#              'ring-1', 'ring-25',
#              '2d_torus-1', '2d_torus-25',
#              'star-1', 'star-25',
#              'random-1', 'random-25']
baselines = ['2d_torus-1', '2d_torus-30',
             'ring-1', 'ring-30',
             'star-1', 'star-30',
             'random-1', 'random-30',
             'complete-1', 'hier_favg', 'ring-based_allreduce']
# baselines = ['ring-based_allreduce', '2d_torus-1', '2d_torus-2',
#              'ring-1', 'ring-2',
#              'star-1', 'star-2',
#              'random-1', 'random-2',
#              'complete-1', 'hier_favg']
baselines = ['ring-based_allreduce', 'hier_favg']
params = {'neighbour-1': {'hops': 1}, 'neighbour-30': {'hops': 1},
          'random-1': {"probability": 0.85}, 'random-30': {"probability": 0.85}}
ps_num = 9
offline_params = {'number': 0, 'time_slot': 2}
fedpma_max_block_duration = 0
underlay = 'geantdistance'
model_size = 31400   # 31400 for LR, 6653480 for CNN  508328, 547544
link_capacity = [1e9]#[1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
node_capacity = [1e9]#[1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
protocol = 'tcp'
wan_latency = 'auto'
client_group_list = [[] for _ in range(ps_num)]
verbose = False
# ####################################################################################################

base_dir = './ns3_rs/%s-%d_offline_%d_%.2f' % (underlay, ps_num, offline_params['number'], offline_params['time_slot'])
time_file_path = os.path.join(base_dir, "ns3-time_%dbytes_%s-%d.json" % (model_size, underlay, ps_num))
os.makedirs(base_dir, exist_ok=True)

rs = OrderedDict({})
for link_capacity, node_capacity in zip(link_capacity, node_capacity):
    key_word = "link=%s,node=%s" % (link_capacity, node_capacity)
    rs[key_word] = {}

    np.random.seed(seed)
    print("############################## link=%.3f Mbps, node=%.3f Mbps ##############################" % (
        link_capacity / 1e6, node_capacity / 1e6))
    np.random.seed(seed)
    network = Network(client_group_list=client_group_list, underlay_name=underlay, parent_dir='.',
                      link_capacity=link_capacity, node_capacity=node_capacity,
                      wan_latency=wan_latency, lan_latency=5e-17, model_size=model_size, verbose=False)
    # network.plot_network('underlay', node_label=True, figsize=(20, 20))

    for iteration in range(num_of_iterations):

        for i, name in enumerate(baselines):
            print("\n" +name)
            start = time.time()
            np.random.seed(seed + iteration)
            my_params = params.get(name, {})

            # if len(my_params) > 0:
            #     params_str = '-'.join(["%s=%s" % (k, v) for k, v in my_params.items()])
            #     method = "%s-%s" % (name, params_str)
            # else:
            #     method = name
            method = name

            save_path = os.path.join(base_dir, "comm_%s.json.%d" % (method, iteration))

            if rs[key_word].get(method) is None:
                rs[key_word][method] = []

            if method == 'ring-based_allreduce':
                time_consumed = network.ring_based_all_reduced(start_time=0, stop_time=None, protocol=protocol,
                                                               verbose=verbose, synchronous=False,
                                                               offline_params=offline_params,
                                                               save_path=save_path)
                rs[key_word][method].append(time_consumed)
                print("---%s: %fs" % (method, time_consumed))
            elif method == 'hier_favg':
                tmp_rs = []
                time_consumed = network.hier_favg(start_time=0, stop_time=None, protocol=protocol, verbose=verbose,
                                                  offline_params=offline_params,
                                                  max_block_duration=None,
                                                  save_path=save_path)
                rs[key_word][method].append(time_consumed)
                print("---%s: %fs" % (method, time_consumed))
            else:
                topo, t = name.split("-")
                network.prepare(topo_name=topo, params=my_params, verbose=True)
                time_consumed = network.pfl_step(times=int(t), start_time=0, stop_time=None, protocol=protocol, port=19,
                                                 verbose=verbose, offline_params=offline_params,
                                                 max_block_duration=fedpma_max_block_duration,
                                                 save_path=save_path)
                rs[key_word][method].append(time_consumed)
                print("---%s (c=%s): %fs" % (method, t, time_consumed))
                # network.plot_network('overlay', node_label=True, figsize=(8, 8))

            end = time.time()
            print("simulation time used:", end-start)

        with open(time_file_path, 'w') as f:
            f.write(json.dumps(rs, indent=4))

    del network

print("ps num: ", ps_num)
print("link=%.3f Mbps, node=%.3f Mbps" % (link_capacity / 1e6, node_capacity / 1e6))
print(rs)

# with open(time_file_path, 'w') as f:
#     f.write(json.dumps(rs, indent=4))
