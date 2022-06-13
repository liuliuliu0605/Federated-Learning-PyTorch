import skfuzzy

from .node import *
from .routing_protocol import *
from .utils import *

import config as cf
import logging

"""Every node communicate its position to the base station. Then the 
BS uses FCM to define clusters and broadcast this information to the
network. Finally, a round is executed.
"""


class FCM(RoutingProtocol):

    def initialize(self, network):
        """The base station uses Fuzzy C-Means to clusterize the network. The
        optimal number of clusters is calculated. Then FCM is used to select
        the heads (centroids) for each cluster (only in the initial round).
        Then each cluster head chooses a new cluster head for the next round.
        该算法使用的是模糊C-Mean算法进行聚类
        路由算法是在最接近聚类中心的节点中选取剩余能量最多的作为簇头节点
        其他普通节点通过簇头节点再发送给基站
        Referece:
          D. C. Hoang, R. Kumar and S. K. Panda, "Fuzzy C-Means clustering
          protocol for Wireless Sensor Networks," 2010 IEEE International
          Symposium on Industrial Electronics, Bari, 2010, pp. 3477-3482.
        """
        nb_clusters = cf.NB_CLUSTERS

        # format data to shape expected by skfuzzy API
        data = [[node.pos_x, node.pos_y] for node in network[0:-1]]
        data = np.array(data).transpose()
        centroids, membership = skfuzzy.cluster.cmeans(data, nb_clusters,
                                                       cf.FUZZY_M, error=0.005,
                                                       maxiter=1000,
                                                       init=None)[0:2]  # [0:2]表示只取API前3个数据
        # assign node nearest to centroid as cluster head
        heads = []
        # also annotates centroids to network
        network.centroids = []
        for cluster_id, centroid in enumerate(centroids):
            tmp_centroid = Node(0)
            tmp_centroid.pos_x = centroid[0]
            tmp_centroid.pos_y = centroid[1]
            network.centroids.append(tmp_centroid)  # FCM确定的聚类中心
            nearest_node = None
            shortest_distance = cf.INFINITY
            for node in network[0:-1]:
                distance = calculate_distance(node, tmp_centroid)
                # 计算与聚类中心最近的节点
                if distance < shortest_distance:
                    nearest_node = node
                    shortest_distance = distance
            # 选择最近聚类中心节点为簇头
            nearest_node.next_hop = cf.BSID
            nearest_node.membership = cluster_id  # 标记为该类群
            heads.append(nearest_node)

        # assign ordinary network to cluster heads using fcm
        for i, node in enumerate(network[0:-1]):
            if node in heads:  # node is already a cluster head 过滤簇头节点
                continue
            cluster_id = np.argmax(membership[:, i])
            node.membership = cluster_id
            head = [x for x in heads if x.membership == cluster_id][0]  # 普通节点选择该类群的簇头作为下一跳
            node.next_hop = head.id

        users_groups = [[] for _ in range(nb_clusters)]
        for i, node in enumerate(network[0:cf.BSID]):
            users_groups[node.membership].append(node.id)
        for i, users_group in enumerate(users_groups):
            logging.debug('FCM cluster %d: %s' % (i, users_group))
        logging.debug('FCM head initialization: %s' % [header.id for header in heads])

    def setup_phase(self, network, round_nb):
        self.head_rotation(network)

    def head_rotation(self, network):
        # head rotation
        # current cluster heads choose next cluster head with the most
        # residual energy and nearest to the cluster centroid
        # 当前簇头选举：选择剩余能量最大的节点
        next_header_ids = []
        for cluster_id in range(0, cf.NB_CLUSTERS):
            cluster = network.get_nodes_by_membership(cluster_id)
            # check if there is someone alive in this cluster
            if len(cluster) == 0:
                continue

            # someone is alive, find node with highest energy in the cluster
            # to be the next cluster head
            highest_energy = cf.MINUS_INFINITY
            next_head = None
            for node in cluster:
                if node.energy_source.energy > highest_energy:
                    highest_energy = node.energy_source.energy
                    next_head = node

            for node in cluster:
                node.next_hop = next_head.id
            next_head.next_hop = cf.BSID
            next_header_ids.append(next_head.id)
            # print("**", [(node.id, node.membership, node.is_head(), node._next_hop, node.energy_source.energy) for node in cluster])
        logging.debug('FCM head rotation: %s' % next_header_ids)