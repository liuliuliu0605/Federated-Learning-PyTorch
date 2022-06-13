import numpy as np
from src.simulator.internet.network import Network
import warnings
from collections import OrderedDict
import json
import ns.core
warnings.filterwarnings("ignore")

seed = 123456

# ####################################################################################################
topo = ['hier_favg'] # ['hier_favg', 'complete', 'ring-based allreduce', 'neighbour', 'ring', '2d_torus']
params = [{}, {}, {}, {'hops': 1}, {}, {}]
ps_num = 9
offline_params = {'number': 1, 'time_slot': 0.5}
fedpma_max_block_duration = 0
underlay = 'geantdistance'
max_times = 6
times = [[1], [1], [1], range(1, max_times+1), range(1, max_times+1), range(1, max_times+1)]
model_size = 100  # 31400 for LR, 6653480 for CNN
link_capacity = [1e9]  # [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
node_capacity = [1e9]  # [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
protocol = 'tcp'
wan_latency = 'auto'
client_group_list = [[] for _ in range(ps_num)]
# ####################################################################################################


file_name = "ns3-time_%dbytes_%s-%d.json" % (model_size, underlay, ps_num)
rs = {}
for capacity_index in range(len(link_capacity)):
    key_word = str(link_capacity[capacity_index])
    rs[key_word] = OrderedDict()

    np.random.seed(seed)
    print("******link=%.3f Mbps, node=%.3f Mbps******" % (link_capacity[capacity_index]/1e6, node_capacity[capacity_index]/1e6))
    network = Network(client_group_list=client_group_list, underlay_name=underlay, parent_dir='.',
                      link_capacity=link_capacity[capacity_index], node_capacity=node_capacity[capacity_index],
                      wan_latency=wan_latency, lan_latency=5e-17, model_size=model_size, verbose=False)
    network.plot_network('underlay', node_label=True, figsize=(20, 20))

    for i, t in enumerate(topo):
        np.random.seed(seed)
        params_str = '-'.join(["%s=%s"%(k,v) for k, v in params[i].items()])
        if len(params[i]) > 0:
            method = "%s-%s" % (t, params_str)
        else:
            method = t

        if t == 'ring-based allreduce':
            tmp_rs = []
            time_consumed = network.ring_based_all_reduced(start_time=0, stop_time=None, protocol=protocol,
                                                           verbose=False, synchronous=False, offline_params=offline_params)
            print("\n---%s: %fs" % (method, time_consumed))
            # tmp_rs.append(time_consumed)
            # time_consumed = network.ring_based_all_reduced(start_time=0, stop_time=None, protocol=protocol,
            #                                                verbose=False, synchronous=True)
            # print("ring allreduce:%f s" % (time_consumed))
            tmp_rs.append(time_consumed)
            rs[key_word][method] = tmp_rs
        elif t == 'hier_favg':
            tmp_rs = []
            time_consumed = network.hier_favg(start_time=0, stop_time=None, protocol=protocol, verbose=False,
                                              offline_params = offline_params, max_block_duration=fedpma_max_block_duration)
            print("\n---%s: %fs" % (method, time_consumed))
            tmp_rs.append(time_consumed)
            rs[key_word][method] = tmp_rs
        else:
            tmp_rs = []
            for tt in times[i]:
                network.prepare(topo_name=t, params=params[i], verbose=True)
                time_consumed = network.pfl_step(times=tt, start_time=0, stop_time=None, protocol=protocol, port=19,
                                                 verbose=False, offline_params=offline_params,
                                                 max_block_duration=fedpma_max_block_duration)
                tmp_rs.append(time_consumed)
                print("\n---%s (c=%d): %fs" % (method, tt, time_consumed))
            rs[key_word][method] = tmp_rs
            network.plot_network('overlay', node_label=True, figsize=(8, 8))
            # network.plot_flow_stat()

    del network

print("ps num: ", ps_num)
print("link=%.3f Mbps, node=%.3f Mbps" % (link_capacity[capacity_index]/1e6, node_capacity[capacity_index]/1e6))
print(rs)

# with open("./ns3_rs/%s" % file_name, 'w') as f:
#     f.write(json.dumps(rs, indent=4))
