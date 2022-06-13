import sys
from sklearn.datasets import make_blobs

from src.simulator.wsn.network import Network
from src.simulator.wsn.utils import *
from src.simulator.wsn.fcm import *
from src.simulator.wsn.direct_communication import *
from src.utils import complete, star

seed = 1
np.random.seed(seed )

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

traces = {}

topo = complete(cf.NB_CLUSTERS)
# topo = independent(cf.NB_CLUSTERS)
# topo = star(cf.NB_CLUSTERS)
# topo = ring(cf.NB_CLUSTERS)


centers = [[50, 225], [25, 110], [125, 20], [220, 80], [200, 225]]
X, y = make_blobs(n_samples=100, centers=centers, n_features=2,
                  random_state=seed, cluster_std=15)
traces = {}
network = Network(init_nodes=X, topo=topo)
# network = Network(topo=topo)

for routing_topology in ['FCM']:#, 'DC']:
    network.reset()
    routing_protocol_class = eval(routing_topology)
    network.init_routing_protocol(routing_protocol_class())

    # traces[routing_topology] = network.simulate()
    for i in range(1000):
        print("--------Round %d--------"% i)
        network.activate_mix()
        traces[routing_topology] = network.simulate_one_round()
        network.deactivate_mix()
        if len(network.get_alive_nodes()) == 0 :
            break
    # plot_clusters(network)
    # plot_time_of_death(network)
    # print(network.energy_dis)
    # print(network.energy_dis['inter-comm']/ network.energy_dis['intra-comm'])
    print("All death round: ", i)
    print("First death round: ", network.first_depletion)
    print("Energy:", network.energy_dis)

plot_traces(traces)