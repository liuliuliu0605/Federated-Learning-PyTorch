from .node import *
from .tracer import *
import logging


class Network(list):
    """This class stores a list with all network nodes plus the base sta-
    tion. Its methods ensure the network behavior.
    """

    def __init__(self, init_nodes=None, topo=None, num_nodes=100):
        logging.debug('Instantiating nodes...')
        if init_nodes is not None:
            nodes = [Node(i, self, loc) for i, loc in enumerate(init_nodes)]
            self.extend(nodes)
        else:
            nodes = [Node(i, self) for i in range(0, cf.NB_NODES)]
            self.extend(nodes)
        # last node in nodes is the base station
        base_station = Node(cf.BSID, self)
        base_station.pos_x = cf.BS_POS_X
        base_station.pos_y = cf.BS_POS_Y
        self.append(base_station)

        self._dict = {}
        for node in self:
            self._dict[node.id] = node

        self.perform_two_level_comm = 1
        self.round = 0
        self.centroids = []
        self.routing_protocol = None
        self.sleep_scheduler_class = None

        self.initial_energy = self.get_remaining_energy()
        self.first_depletion = 0
        self.per30_depletion = 0
        self.energy_spent = []

        self.deaths_this_round = 0
        self.all_alive = 1
        self.percent70_alive = 1
        self.tracer = Tracer()
        self.topo = topo
        self.energy_dis = {'intra-comm': 0, 'inter-comm': 0, 'local-update': 0, 'default': 0}
        self.is_mix = False

    def init_routing_protocol(self, routing_protocol):
        self.routing_protocol = routing_protocol
        self.routing_protocol.initialize(self)

    def reset(self):
        """Set nodes to initial state so the same placement of nodes can be
        used by different techniques.
        """
        for node in self:
            node.energy_source.recharge()
            node.reactivate()

        # allows for updates of BS position between simulations
        self[-1].pos_x = cf.BS_POS_X
        self[-1].pos_y = cf.BS_POS_Y

        self.round = 0
        self.centroids = []
        self.energy_spent = []

        self.routing_protocol = None
        self.sleep_scheduler_class = None

        self.first_depletion = 0
        self.per30_depletion = 0

        self.deaths_this_round = 0
        self.all_alive = 1
        self.percent70_alive = 1
        self.tracer = Tracer()
        self.energy_dis = {'intra-comm': 0, 'inter-comm': 0, 'local-update': 0, 'default': 0}

    def set_topo(self, topo):
        self.topo = topo

    def activate_mix(self):
        self.is_mix = True

    def deactivate_mix(self):
        self.is_mix = False

    def simulate_one_round(self):
        # print_args = (self.round, self.get_remaining_energy())
        # print("round %d: total remaining energy: %f" % print_args)
        nb_alive_nodes = self.count_alive_nodes()
        if nb_alive_nodes == 0:
            return self.tracer
        self.tracer['alive_nodes'][2].append(nb_alive_nodes)
        nb_alive_clusters = self.count_alive_clusters()
        self.tracer['alive_clusters'][2].append(nb_alive_clusters)
        if cf.TRACE_ENERGY:
            self.tracer['energies'][2].append(self.get_remaining_energy())

        self.routing_protocol.setup_phase(self, self.round)

        # check if someone died
        if self.deaths_this_round != 0:
            if self.all_alive == 1:
                self.all_alive = 0
                self.first_depletion = self.round
                self.tracer['first_depletion'][2].append(self.first_depletion)
            if float(nb_alive_nodes) / float(cf.NB_NODES) < 0.7 and \
                    self.percent70_alive == 1:
                self.percent70_alive = 0
                self.per30_depletion = self.round
                self.tracer['30per_depletion'][2].append(self.per30_depletion)

        # clears dead counter
        self.deaths_this_round = 0

        # communication round in federated learning
        self._run_round_fl()
        self.round += 1

        return self.tracer

    def simulate(self):
        tracer = Tracer()

        all_alive = 1
        percent70_alive = 1
        self.deaths_this_round = 0

        # 开始循环迭代计算
        for round_nb in range(0, cf.MAX_ROUNDS):
            self.round = round_nb
            # print_args = (round_nb, self.get_remaining_energy())
            # print("round %d: total remaining energy: %f" % print_args)
            nb_alive_nodes = self.count_alive_nodes()

            if nb_alive_nodes == 0:
                break
            tracer['alive_nodes'][2].append(nb_alive_nodes)
            if cf.TRACE_ENERGY:
                tracer['energies'][2].append(self.get_remaining_energy())

            self.routing_protocol.setup_phase(self, round_nb)

            # check if someone died
            if self.deaths_this_round != 0:
                if all_alive == 1:
                    all_alive = 0
                    self.first_depletion = round_nb
                if float(nb_alive_nodes) / float(cf.NB_NODES) < 0.7 and \
                        percent70_alive == 1:
                    percent70_alive = 0
                    self.per30_depletion = round_nb

            # clears dead counter
            self.deaths_this_round = 0

            # communication round in federated learning
            self._run_round_fl()

        tracer['first_depletion'][2].append(self.first_depletion)
        tracer['30per_depletion'][2].append(self.per30_depletion)

        return tracer

    def _run_round_fl(self):
        before_energy = self.get_remaining_energy()
        self._broadcast_models()
        self._local_updates(cf.LOCAL_UPDATES)
        self._gather_models()
        if self.is_mix:
            self._mix_models()
        after_energy = self.get_remaining_energy()
        self.energy_spent.append(before_energy - after_energy)

    def _local_updates(self, k=1):
        # node performs local updates
        logging.debug('Performing local updates...')
        for node in self.get_alive_nodes():
            node.update(cf.MSG_LENGTH, k)

    def _broadcast_models(self):
        # cluster heads distribute models to members
        logging.debug('Broadcasting models...')

        heads = self.get_heads()
        for head in heads:
            members = self.get_ordinary_nodes_by_membership(head.membership)
            head.broadcast(msg_length=cf.MSG_LENGTH, destination_list=members, type='intra-comm')

        # direct communication with BS
        if len(heads) == 0:
            members = self.get_nodes_by_membership(cf.BSID)
            self[cf.BSID].broadcast(msg_length=cf.MSG_LENGTH, destination_list=members, type='intra-comm')

    def _gather_models(self):
        # cluster members return model to cluster headers
        logging.debug('Gathering models...')
        cluster_members = self.get_ordinary_nodes()
        for node in cluster_members:
            node.transmit(cf.MSG_LENGTH, type='intra-comm')

        # direct communication with BS
        if len(cluster_members) == 0:
            for node in self.get_nodes_by_membership(cf.BSID):
                node.transmit(cf.MSG_LENGTH, type='intra-comm')

    def _mix_models(self):
        # cluster headers mix with each other
        logging.debug('Mixing models...')
        for node in self.get_heads():

            # broadcast between cluster headers
            other_headers = []
            for membership, w in enumerate(self.topo[node.membership]):
                if membership != node.membership and w > 0:
                    other_header = self.get_header_by_membership(membership)
                    if other_header is not None:
                        other_headers.append(other_header)
            if len(other_headers) > 0:
                node.broadcast(msg_length=cf.MSG_LENGTH, destination_list=other_headers, type='inter-comm')

            # unicast between cluster headers
            # for membership, w in enumerate(self.topo[node.membership]):
            #   if membership != node.membership and w > 0:
            #     other_header = self.get_header_by_membership(membership)
            #     if other_header is not None:
            #       node.transmit(msg_length=cf.MSG_LENGTH, destination=other_header)

    def get_alive_nodes(self):
        """Return nodes that have positive remaining energy."""
        return [node for node in self[0:-1] if node.alive]

    def get_alive_clusters(self):
        """Return nodes that have positive remaining energy."""
        return list(set([x.membership for x in self[:-1] if x.alive]))

    def get_cluster_members(self):
        return [[node.id for node in self if node.membership == membership] for membership in range(cf.NB_CLUSTERS)]

    def get_active_nodes(self):
        """Return nodes that have positive remaining energy and that are
        awake."""
        is_active = lambda x: x.alive and not x.is_sleeping
        return [node for node in self[0:-1] if is_active(node)]

    def get_ordinary_nodes(self):
        return [node for node in self if node.is_ordinary() and node.alive]

    def get_ordinary_nodes_by_membership(self, membership):
        return [node for node in self if node.is_ordinary() and node.alive
                and node.membership == membership]

    def get_heads(self, only_alives=1):
        input_set = self.get_alive_nodes() if only_alives else self
        return [node for node in input_set if node.is_head() and node.membership != cf.BSID]

    def get_sensor_nodes(self):
        """Return all nodes except base station."""
        return [node for node in self[0:-1]]

    def get_average_energy(self):
        return np.average(self.energy_spent)

    def someone_alive(self):
        """Finds if there is at least one node alive. It excludes the base station,
           which is supposed to be always alive."""
        for node in self[0:-1]:
            if node.alive == 1:
                return 1
        return 0

    def count_alive_nodes(self):
        return sum(x.alive for x in self[:-1])

    def count_alive_clusters(self):
        return len(set([x.membership for x in self[:-1] if x.alive]))

    def get_BS(self):
        # intention: make code clearer for non-Python readers
        return self[-1]

    def get_node(self, id):
        """By default, we assume that the id is equal to the node's posi-
        tion in the list, but that may not be always the case.
        """
        return self._dict[id]

    def get_nodes_by_membership(self, membership, only_alives=1):
        """Returns all nodes that belong to this membership/cluster."""
        input_set = self.get_alive_nodes() if only_alives else self
        condition = lambda node: node.membership == membership and node.id != cf.BSID
        return [node for node in input_set if condition(node)]

    def get_header_by_membership(self, membership, only_alives=1):
        input_set = self.get_alive_nodes() if only_alives else self
        condition = lambda node: node.membership == membership and node.id != cf.BSID and node.is_head()
        header = [node for node in input_set if condition(node)]
        if len(header) == 1:
            return header[0]
        else:
            return None

    def get_remaining_energy(self, ignore_nodes=None):
        """Returns the sum of the remaining energies at all nodes.
        计算整个网络节点剩余能量
        """
        set = self.get_alive_nodes()
        if len(set) == 0:
            return 0
        if ignore_nodes:
            set = [node for node in set if node not in ignore_nodes]
        transform = lambda x: x.energy_source.energy  # 通过匿名函数Lambda传递节点能量
        energies = [transform(x) for x in set]
        return sum(x for x in energies)

    def split_in_clusters(self, nb_clusters=cf.NB_CLUSTERS):
        """Split this nodes object into other nodes objects that contain only
        information about a single cluster."""
        clusters = []
        for cluster_idx in range(0, nb_clusters):
            nodes = self.get_nodes_by_membership(cluster_idx)
            cluster = Network(init_nodes=nodes)
            cluster.append(self.get_BS())
            clusters.append(cluster)
        return clusters

