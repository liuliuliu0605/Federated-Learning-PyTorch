import ns.core
import numpy as np

from sender import Sender
from receiver import Receiver

class RingAllReduce:

    def __init__(self, model_size, ps_node_container, ring_list, data_rate=1e9, packet_size=536,
                 protocol='tcp', port=9, verbose=False):
        assert protocol in ['tcp', 'udp']
        assert ps_node_container.GetN() == len(ring_list)
        # make sure it is a ring, TODO

        self.model_size = model_size
        self.ps_node_container = ps_node_container
        self.ring_list = ring_list
        self.data_rate = data_rate
        self.packet_size = packet_size

        self.protocol = protocol
        self.port = port

        self.node_num = len(ring_list)
        self.verbose = verbose

        self._time_consuming = 0
        self._sink_socket_list = []
        self._source_socket_list = []
        self._app_sender_list = []
        self._app_receiver_list = []
        self._resource_dict = [{"sender": [], "receiver": []} for _ in range(self.node_num)]

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
        self._resource_dict = [{"sender": [], "receiver": []} for _ in range(self.node_num)]

    def update_phase(self):
        total_bytes_rcv_this_phase = self.model_size_per_phase * self.phase
        self.finished = True
        for rcv_bytes in self.total_rcv_bytes_list:
            if rcv_bytes < total_bytes_rcv_this_phase:
                self.finished = False
                break
        if self.finished:
            self.phase += 1
            if self.phase > 2 * (self.node_num - 1):
                if self.verbose:
                    ns.core.Simulator.Stop()
                    print("Each PS totally received bytes:")
                    print(self.total_rcv_bytes_list)
                    print("Allreduce done!")
                return
            # notify all pses enter the second phase
            if self.verbose:
                print("Enter phase %d" % self.phase)
            for i in range(self.node_num):
                src_node = self.ps_node_container.Get(self.ring_list[i])
                dst_node = self.ps_node_container.Get(self.ring_list[(i + 1) % self.node_num])
                if self.verbose:
                    print("Flow: %d -> %d" % (self.ring_list[i], self.ring_list[(i + 1) % self.node_num]))
                apps_sender = self.send_packets(src_node, dst_node)

    def set_sink_nodes(self, start_time=0, stop_time=None):
        for i in range(self.node_num):
            sink_node = self.ps_node_container.Get(i)
            sink_socket = ns.network.Socket.CreateSocket(sink_node,
                                                         ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                     format(self.protocol.capitalize())))
            sink_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), self.port)

            app_receiver = Receiver()
            app_receiver.Setup(sink_socket, sink_address, self.rx_size_per_phase, protocol=self.protocol,
                               verbose=self.verbose)
            sink_node.AddApplication(app_receiver)
            app_receiver.SetStartTime(ns.core.Seconds(start_time))
            if stop_time is not None:
                app_receiver.SetStopTime(ns.core.Seconds(stop_time))
            self._sink_socket_list.append(sink_socket)
            self._app_receiver_list.append(app_receiver)

    def set_source_nodes(self, start_time=0, stop_time=None):
        if self.verbose:
            print("Ring flow: " + '->'.join(self.ring_list + self.ring_list[0]))

        for i in range(self.node_num):
            src = self.ring_list[i]
            dst = self.ring_list[(i+1)%self.node_num]

            src_node = self.ps_node_container.Get(src)
            dst_node = self.ps_node_container.Get(dst)

            ipv4 = dst_node.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_int_addr = ipv4.GetAddress(1, 0)
            ip_addr = ipv4_int_addr.GetLocal()
            sink_address = ns.network.InetSocketAddress(ip_addr, self.port)

            source_socket = ns.network.Socket.CreateSocket(src_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                format(self.protocol.capitalize())))

            apps_sender = Sender()
            apps_sender.Setup(source_socket, sink_address, self.packet_size, self.rx_size_per_phase,
                      ns.network.DataRate("{:f}bps".format(self.data_rate)), verbose=self.verbose)
            src_node.AddApplication(apps_sender)
            apps_sender.SetStartTime(ns.core.Seconds(start_time))
            if stop_time is not None:
                apps_sender.SetStopTime(ns.core.Seconds(stop_time))
            self._source_socket_list.append(source_socket)
            self._app_sender_list.append(apps_sender)

    def get_time_consuming(self):
        return self._time_consuming

    def run(self, start_time, stop_time):
        self.reset()
        self.set_sink_nodes(start_time, stop_time)
        self.set_source_nodes(start_time, stop_time)

        for i in range(self.node_num):
            src_node = self.ps_node_container.Get(self.ring_list[i])
            dst_node = self.ps_node_container.Get(self.ring_list[(i+1)%self.node_num])
            if self.verbose:
                print("Flow: %d -> %d" % (self.ring_list[i], self.ring_list[(i+1)%self.node_num]))
            apps_sender = self.send_packets(src_node, dst_node)
            apps_sender.Start(ns.core.Seconds(start_time))
            if stop_time is not None:
                apps_sender.Stop(ns.core.Seconds(stop_time))


