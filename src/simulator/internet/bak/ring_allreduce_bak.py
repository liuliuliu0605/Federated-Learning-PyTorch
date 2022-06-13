import ns.core


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

        self.node_id_index_dict = {self.ps_node_container.Get(i).GetId(): i for i in range(self.node_num)}
        self.model_size_per_phase = int(self.model_size / self.node_num)

        self._time_consuming = 0
        self._sink_socket_list = []
        self._source_socket_list = []
        self._app_sender_list = []
        self._app_receiver_list = []

        self.phase = 1
        self.total_rcv_bytes_list = [0 for _ in range(self.node_num)]
        self.current_time = 0
        self.finished = False

    def reset(self):
        self._time_consuming = 0
        for sink_socket in self._sink_socket_list:
            sink_socket.Close()
        self._sink_socket_list.clear()
        for source_socket in self._source_socket_list:
            source_socket.Close()
        self._source_socket_list.clear()
        self._app_sender_list.clear()
        self._app_sender_list.clear()



        self.phase = 1
        self.total_rcv_bytes_list = [0 for _ in range(self.node_num)]
        self.current_time = 0
        for sink_socket in self._sink_socket_list:
            sink_socket.Close()
        self._sink_socket_list.clear()
        self.finished = False

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

    def set_sink_nodes(self):

        def rcv_packet(socket):
            src = ns.network.Address()
            while True:
                packet = socket.RecvFrom(1024, 0, src)
                if packet is None or packet.GetSize() <= 0:
                    break

                index = self.node_id_index_dict[socket.GetNode().GetId()]
                self.total_rcv_bytes_list[index] += packet.GetSize()

                rcv_time = ns.core.Simulator.Now().GetSeconds()
                if rcv_time > self.current_time:
                    self.current_time = rcv_time
                if self.verbose and ns.network.InetSocketAddress.IsMatchingType(src):
                    address = ns.network.InetSocketAddress.ConvertFrom(src)
                    print("At time %.6f packet sink received %d bytes from %s port %d" %
                          (rcv_time, packet.GetSize(), address.GetIpv4(), address.GetPort()))

                self.update_phase()

        def accept_callback(a, b):
            return True

        def new_connection(socket, address):
            socket.SetRecvCallback(rcv_packet)

        # create sink application for PSes
        for i in range(self.node_num):
            sink_node = self.ps_node_container.Get(i)
            sink_socket = ns.network.Socket.CreateSocket(sink_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(self.protocol.capitalize())))
            if self.protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
            else:
                sink_socket.SetRecvCallback(rcv_packet)

            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), self.port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

            self._sink_socket_list.append(sink_socket)

    def init_app(self, start_time, stop_time):
        if self.verbose:
            print("Enter phase %d" % self.phase)

        self.set_sink_nodes()

        for i in range(self.node_num):
            src_node = self.ps_node_container.Get(self.ring_list[i])
            dst_node = self.ps_node_container.Get(self.ring_list[(i+1)%self.node_num])
            if self.verbose:
                print("Flow: %d -> %d" % (self.ring_list[i], self.ring_list[(i+1)%self.node_num]))
            apps_sender = self.send_packets(src_node, dst_node)
            apps_sender.Start(ns.core.Seconds(start_time))
            if stop_time is not None:
                apps_sender.Stop(ns.core.Seconds(stop_time))

    def send_packets(self, src_node, dst_node):
        ipv4 = dst_node.GetObject(ns.internet.Ipv4.GetTypeId())
        ipv4_int_addr = ipv4.GetAddress(1, 0)
        ip_addr = ipv4_int_addr.GetLocal()

        sender = ns.applications.BulkSendHelper(
            "ns3::{:s}SocketFactory".format(self.protocol.capitalize()),
            ns.network.Address(ns.network.InetSocketAddress(ip_addr, self.port)))
        sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size_per_phase))
        apps_sender = sender.Install(src_node)

        return apps_sender

    def get_current_time(self):
        return self.current_time


