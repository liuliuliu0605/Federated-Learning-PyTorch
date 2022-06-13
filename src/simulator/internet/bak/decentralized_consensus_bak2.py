import ns.core
import numpy as np

from src.simulator.internet.sender import Sender
from src.simulator.internet.receiver import Receiver


class DecentralizedConsensus:

    def __init__(self, model_size, ps_node_container, matrix, data_rate=1e9, packet_size=536,
                 protocol='tcp', port=9, verbose=False, offline=0.0):
        assert protocol in ['tcp', 'udp']
        assert ps_node_container.GetN() == len(matrix)

        self.model_size = model_size
        self.ps_node_container = ps_node_container
        self.matrix = matrix
        self.data_rate = data_rate
        self.packet_size = packet_size

        self.protocol = protocol
        self.port = port

        self.node_num = self.ps_node_container.GetN()
        self.verbose = verbose
        self.offline = offline

        self._time_consuming = 0
        self._sink_socket_list = []
        self._source_socket_list = []
        self._app_sender_list = []
        self._app_receiver_list = []
        self._resource_dict = [{"sender": [], "receiver":[]} for _ in range(self.node_num)]

    def reset(self):
        self._time_consuming = 0
        for sink_socket in self._sink_socket_list:
            sink_socket.Close()
        self._sink_socket_list.clear()
        for source_socket in self._source_socket_list:
            source_socket.Close()
        self._source_socket_list = []
        self._app_sender_list = []
        self._app_sender_list = []
        self._resource_dict = [{"sender": [], "receiver":[]} for _ in range(self.node_num)]

    def set_sink_nodes(self, start_time=0, stop_time=None, phases=1):
        for i in range(self.node_num):
            phase_rx_size = (np.count_nonzero(self.matrix[:, i] > 0) - 1) * self.model_size
            if phase_rx_size <= 0:
                continue

            sink_node = self.ps_node_container.Get(i)
            sink_socket = ns.network.Socket.CreateSocket(sink_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(self.protocol.capitalize())))
            sink_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), self.port)

            app_receiver = Receiver()

            app_receiver.Setup(sink_socket, sink_address, phase_rx_size, phases=phases,
                               protocol=self.protocol, verbose=self.verbose, id=i, offline=self.offline)
            sink_node.AddApplication(app_receiver)
            app_receiver.SetStartTime(ns.core.Seconds(start_time))
            if stop_time is not None:
                app_receiver.SetStopTime(ns.core.Seconds(stop_time))
            self._sink_socket_list.append(sink_socket)
            self._app_receiver_list.append(app_receiver)
            self._resource_dict[i]['receiver'].append(len(self._app_receiver_list) - 1)

    def set_source_nodes(self, start_time=0, stop_time=None, phases=1):
        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.matrix[src, dst] > 0:

                    if self.verbose:
                        print("Mixing model: PS %d -> PS %d" % (src, dst))

                    src_node = self.ps_node_container.Get(src)
                    dst_node = self.ps_node_container.Get(dst)

                    ipv4 = dst_node.GetObject(ns.internet.Ipv4.GetTypeId())
                    ipv4_int_addr = ipv4.GetAddress(1, 0)
                    ip_addr = ipv4_int_addr.GetLocal()
                    sink_address = ns.network.InetSocketAddress(ip_addr, self.port)

                    source_socket = ns.network.Socket.CreateSocket(src_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                        format(self.protocol.capitalize())))

                    apps_sender = Sender()
                    apps_sender.Setup(source_socket, sink_address, self.packet_size, self.model_size,
                              ns.network.DataRate("{:f}bps".format(self.data_rate)), phases=phases,
                                      verbose=self.verbose, id=src)
                    src_node.AddApplication(apps_sender)
                    apps_sender.SetStartTime(ns.core.Seconds(start_time))
                    if stop_time is not None:
                        apps_sender.SetStopTime(ns.core.Seconds(stop_time))
                    self._source_socket_list.append(source_socket)
                    self._app_sender_list.append(apps_sender)
                    self._resource_dict[src]['sender'].append(len(self._app_sender_list) - 1)

                    # tell receiver the upstream sender
                    self._app_receiver_list[dst].add_upstream_senders(apps_sender)

    def combine_local_senders_recievers(self):
        for resource in self._resource_dict:
            senders = [self._app_sender_list[index] for index in resource['sender']]

            # senders without receivers will transmit data without blocking
            for sender in senders:
                sender.associate_local_receivers([self._app_receiver_list[index] for index in resource['receiver']])

            # receivers will awake senders until receiving enough data in some phase
            if len(resource['receiver']) > 0:
                receiver = self._app_receiver_list[resource['receiver'][0]]
                receiver.associate_local_senders(senders)

    # def associate_receivers_with_upstream_senders(self):
    #     for j, resource in enumerate(self._resource_dict):
    #         if len(resource['receiver']) > 0:
    #             receiver = self._app_receiver_list[resource['receiver'][0]]
    #             upstream_senders = []
    #             src_array = np.nonzero(self.matrix[:, j])[0]
    #             src_array = src_array[src_array != j]
    #             for src in src_array:
    #                 upstream_senders += [self._app_sender_list[index] for index in self._resource_dict[src]['sender']]
    #             receiver.associate_upstream_senders(upstream_senders)

    def get_time_consuming(self):
        return self._time_consuming

    def is_finshied(self):
        sender_states = [sender.is_finished() for sender in self._app_sender_list]
        receiver_states = [sender.is_finished() for sender in self._app_sender_list]
        return np.all(sender_states) and np.all(receiver_states)

    def run(self, start_time, stop_time, phases=1):
        self.reset()

        self.set_sink_nodes(start_time, stop_time, phases)
        self.set_source_nodes(start_time, stop_time, phases)
        self.combine_local_senders_recievers()

        start_of_simulation = ns.core.Simulator.Now().GetSeconds() + start_time
        ns.core.Simulator.Run()
        end_of_simulation = max([app_receiver.get_current_time() for app_receiver in self._app_receiver_list])
        if self.is_finshied():
            self._time_consuming = end_of_simulation - start_of_simulation
        else:
            self._time_consuming = -1
        return self._time_consuming

