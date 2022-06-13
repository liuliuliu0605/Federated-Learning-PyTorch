import ns.core
import numpy as np
# from multiprocessing import Lock, Value


class DecentralizedConsensus:

    def __init__(self, model_size, ps_node_container, matrix, protocol='tcp', port=9, verbose=False):
        assert protocol in ['tcp', 'udp']
        assert ps_node_container.GetN() == len(matrix)

        self.ps_node_container = ps_node_container
        self.matrix = matrix

        self.node_num = self.ps_node_container.GetN()
        self.model_size = model_size
        self.protocol = protocol
        self.port = port
        self.verbose = verbose
        self.node_id_index_dict = {self.ps_node_container.Get(i).GetId(): i for i in range(self.node_num)}
        self.expected_rcv_bytes_list = [(np.count_nonzero(row > 0) - 1) * self.model_size for row in self.matrix]
        self.total_rcv_bytes_list = [0 for _ in range(self.node_num)]
        # self.total_rcv_bytes_list = [Value('i', 0) for _ in range(self.node_num)]

        # self.counter = Value('i', 0)
        # self.lock = Lock()

        self.current_time = 0
        self._sink_socket_list = []
        self.finished = False

        ns.core.Config.SetDefault("ns3::TcpSocket::SndBufSize", ns.core.UintegerValue(32768000))
        ns.core.Config.SetDefault("ns3::TcpSocket::RcvBufSize", ns.core.UintegerValue(32768000))
        # ns.core.Config.SetDefault("ns3::TcpSocket::SegmentSize", ns.core.UintegerValue(500))

        # ns.core.Config.SetDefault("ns3::FqCoDelQueueDisc::Packet limit", ns.core.UintegerValue(1000))

        ns.core.LogComponentEnable("BulkSendApplication", ns.core.LOG_LOGIC)
        ns.core.LogComponentEnable("BulkSendApplication", ns.core.LOG_DEBUG)

    # def __del__(self):
    #     self.reset()

    def reset(self):
        print("reset!")
        self.total_rcv_bytes_list = [0 for _ in range(self.node_num)]
        self.current_time = 0
        for sink_socket in self._sink_socket_list:
            sink_socket.Close()
        self._sink_socket_list.clear()
        self.finished = False

    def update_phase(self):
        # print(self.total_rcv_bytes_list)
        # print(self.counter.value)
        # print(self.counter.value, self.node_num)
        # print([x.value for x in self.total_rcv_bytes_list])
        # if self.counter.value < self.node_num:
        #     return
        self.finished = True
        for i in range(self.node_num):
            if self.total_rcv_bytes_list[i] < self.expected_rcv_bytes_list[i]:
                self.finished = False
                break
        print(self.total_rcv_bytes_list)
        if self.finished:
            ns.core.Simulator.Stop()
            if self.verbose:
                print("Each PS totally received bytes:")
                print(self.total_rcv_bytes_list)
                print("Mixing done!")
        return

    def set_sink_nodes(self):

        def rcv_packet(socket):
            src = ns.network.Address()
            while True:
                packet = socket.RecvFrom(1458, 0, src)

                if packet is None or packet.GetSize() <= 0:
                    break

                index = self.node_id_index_dict[socket.GetNode().GetId()]
                self.total_rcv_bytes_list[index] += packet.GetSize()
                # with self.lock:
                #     self.total_rcv_bytes_list[index].value += packet.GetSize()

                # if self.total_rcv_bytes_list[index] >= self.expected_rcv_bytes_list[index]:
                #     with self.lock:
                #         self.counter.value += 1

                rcv_time = ns.core.Simulator.Now().GetSeconds()
                if rcv_time > self.current_time:
                    self.current_time = rcv_time
                if self.verbose and ns.network.InetSocketAddress.IsMatchingType(src):
                    address = ns.network.InetSocketAddress.ConvertFrom(src)
                    print("At time %.6f packet sink received %d bytes from %s (%d) port %d" %
                          (rcv_time, packet.GetSize(), address.GetIpv4(), index, address.GetPort()))

                self.update_phase()

        def accept_callback(a, b):
            return True

        def new_connection(socket, address):
            socket.SetRecvCallback(rcv_packet)

        def normal_close(socket):
            print("normal close")

        def error_close(socket):
            print("error close")

        # create sink application for PSes
        for i in range(self.node_num):
            sink_node = self.ps_node_container.Get(i)
            sink_socket = ns.network.Socket.CreateSocket(sink_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(self.protocol.capitalize())))
            if self.protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
                # sink_socket.SetCloseCallbacks(normal_close, error_close)
            else:
                sink_socket.SetRecvCallback(rcv_packet)
            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), self.port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

            self._sink_socket_list.append(sink_socket)


    def init_app(self, start_time, stop_time):
        self.set_sink_nodes()

        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.matrix[src, dst] > 0:

                    if self.verbose:
                        print("Flow: PS %d -> PS %d" % (src, dst))

                    src_node = self.ps_node_container.Get(src)
                    dst_node = self.ps_node_container.Get(dst)

                    apps_sender = self.send_packets(src_node, dst_node)
                    apps_sender.Start(ns.core.Seconds(start_time))
                    if stop_time is not None:
                        apps_sender.Stop(ns.core.Seconds(stop_time))

    def send_packets(self, src_node, dst_node):
        ipv4 = dst_node.GetObject(ns.internet.Ipv4.GetTypeId())
        ipv4_int_addr = ipv4.GetAddress(1, 0)
        ip_addr = ipv4_int_addr.GetLocal()

        if self.protocol == 'tcp':
            sender = ns.applications.BulkSendHelper(
                "ns3::{:s}SocketFactory".format(self.protocol.capitalize()),
                ns.network.Address(ns.network.InetSocketAddress(ip_addr, self.port)))
            sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size))
            apps_sender = sender.Install(src_node)
        else:
            echoClient = ns.applications.UdpEchoClientHelper(ip_addr, self.port)
            echoClient.SetAttribute("MaxPackets", ns.core.UintegerValue(1))
            echoClient.SetAttribute("Interval", ns.core.TimeValue(ns.core.Seconds(0)))
            echoClient.SetAttribute("PacketSize", ns.core.UintegerValue(self.model_size))
            apps_sender = echoClient.Install(src_node)

        # from flow import Flow
        #
        # apps_sender = Flow(self.model_size, write_size=1040, protocol='tcp')
        # src_socket = ns.network.Socket.CreateSocket(src_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
        #                                                          format(self.protocol.capitalize())))
        # apps_sender.start_flow(src_socket, ip_addr, self.port, verbose=False)

        return apps_sender

    def get_current_time(self):
        return self.current_time


