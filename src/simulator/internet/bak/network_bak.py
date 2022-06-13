from utils import *
from matplotlib import pyplot as plt

import ns.core
import ns.network
import ns.internet
import ns.point_to_point
import ns.applications
import ns.mobility
import ns.netanim
import ns.flow_monitor
import ns.wimax
import ns.csma
import ns.uan
import ns.wave

import sys

LAN_LATENCY = 5e-7

class Network:

    def __init__(self, client_group_list, underlay_name='gaia',
                 node_capacity=1e9, link_capacity=1e9,
                 wan_latency='auto', lan_latency=5e-7,
                 model_size=1e3, coord_array=None,
                 parent_dir='simulator/internet', verbose=False, rs_dir='./ns3_rs'):
        """
        Construct underlay network
        """
        # PSes and clients
        self.client_group_list = client_group_list
        self.ps_num = len(client_group_list)
        self.rs_dir = rs_dir
        self.time_consumed_one_step = 0

        # bandwidth params
        self.upload_capacity = int(node_capacity)  # bps
        self.download_capacity = int(node_capacity)  # bps
        self.model_size = int(model_size)  # Bytes
        self.link_capacity = int(link_capacity) # bps
        self.lan_latency = lan_latency
        self.wan_latency = wan_latency

        # get underlay graph and connectivity graph of all nodes
        underlay_dir = os.path.join(parent_dir, 'underlay')
        self._underlay_graph = get_underlay_graph(underlay_name=underlay_name,
                                                  upload_capacity=self.upload_capacity,
                                                  download_capacity=self.download_capacity,
                                                  underlay_dir=underlay_dir,
                                                  latency_assumed=wan_latency)
        self._connectivity_graph = get_connectivity_graph(self._underlay_graph,
                                                          link_capacity=link_capacity,
                                                          latency_assumed=wan_latency)

        # add edge weights to connectivity graph
        for u, v, data in self._connectivity_graph.edges(data=True):
            # weight = data["latency"] + model_size / data["availableBandwidth"]
            weight = data["latency"]
            self._connectivity_graph.add_edge(u, v, weight=weight)

        # get connectivity graph of PSes randomly selected from underlay graph
        assert self.ps_num <= self._connectivity_graph.number_of_nodes()
        self._target_connectivity_graph = get_subset_connectivity_graph(self._connectivity_graph, self.ps_num)
        self._ps_sn_list = list(self._target_connectivity_graph.nodes())

        # initialize nodes in NS3, including PSes and intermediate nodes
        self._nodes, self._ps_clients_nodes_list, self._p2p = self._construct_network(verbose=verbose)

        # allocate node position in NS3
        self._allocate_node_position(coord_array=coord_array, verbose=verbose)

        # simulation monitor
        # ascii = ns.network.AsciiTraceHelper()
        # self._p2p.EnableAsciiAll(ascii.CreateFileStream(os.path.join(self.rs_dir, "network.tr")))
        # self._anim = ns.netanim.AnimationInterface(os.path.join(self.rs_dir, "network.xml"))
        # self._anim.SetMaxPktsPerTraceFile(99999999999999)
        # self._flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
        # self._monitor = self._flowmon_helper.InstallAll()
        # self._monitor = self._flowmon_helper.GetMonitor()
        # self._monitor.SetAttribute("DelayBinWidth", ns.core.DoubleValue(0.01))
        # self._monitor.SetAttribute("JitterBinWidth", ns.core.DoubleValue(0.01))
        # self._monitor.SetAttribute("PacketSizeBinWidth", ns.core.DoubleValue(20))


        self.m_totalRx = 0

    def __del__(self):
        ns.core.Simulator.Destroy()

    def prepare(self, topo_name='ring', verbose=False):
        self._target_overlay_graph = self._topo_formation(topo_name=topo_name)

        self.ps_tx_bytes_list = [0 for _ in range(self.ps_num)]

    def _topo_formation(self, topo_name='ring'):
        if topo_name == 'ring':
            target_overlay_graph = get_ring_overlay(self._target_connectivity_graph)
        elif topo_name == 'complete':
            target_overlay_graph = get_complete_overlay(self._target_connectivity_graph)
        elif topo_name == '2d_torus':
            target_overlay_graph = get_2d_torus_overlay(self._target_connectivity_graph)

        return target_overlay_graph

    def _construct_network(self, verbose=False):

        if verbose:
            print("Create Nodes.")

        # routers
        nodes = ns.network.NodeContainer()
        node_num = self._underlay_graph.number_of_nodes()
        nodes.Create(node_num)

        # clients list
        ps_clients_nodes_list = []
        for i in range(self.ps_num):
            ps_clients_nodes = ns.network.NodeContainer()
            ps_clients_nodes.Create(1+ len(self.client_group_list[i]))
            ps_clients_nodes_list.append(ps_clients_nodes)


        # Install the L3 internet stack on routers and pses.
        if verbose:
            print("Install Internet Stack to Nodes.")
        internet = ns.internet.InternetStackHelper()
        internet.Install(nodes)
        for ps_clients_nodes in ps_clients_nodes_list:
            internet.Install(ps_clients_nodes)

        # create p2p links between routers according to underlay topology
        if verbose:
            print("Create Links Between Routers.")
        p2p = ns.point_to_point.PointToPointHelper()
        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.link_capacity)))

        ipv4_n = ns.internet.Ipv4AddressHelper()
        ipv4_n.SetBase(ns.network.Ipv4Address("76.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))

        linkCount = 0
        for i, j, data in self._underlay_graph.edges(data=True):
            n_links = ns.network.NodeContainer()
            n_links.Add(nodes.Get(i))
            n_links.Add(nodes.Get(j))

            LinkDelay = ns.core.StringValue("{:f}s".format(data['latency']))
            p2p.SetChannelAttribute("Delay", LinkDelay)
            n_devs = p2p.Install(n_links)

            ipv4_n.Assign(n_devs)
            ipv4_n.NewNetwork()
            linkCount += 1
            if verbose:
                print("router [", i, "][", j, "] is physically connected")
        if verbose:
            print("Number of physical links is: ", linkCount)
            print("Number of all routers is: ", nodes.GetN())

        # create csma links between ps and routers
        csma = ns.csma.CsmaHelper()
        csma.SetChannelAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.download_capacity)))
        csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(2e-3)))

        p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("{:f}bps".format(self.download_capacity)))
        p2p.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.Seconds(self.lan_latency)))
        assert len(self._ps_sn_list) <= 253

        for i, sn in enumerate(self._ps_sn_list):

            # set up links between pses and routers, p2p
            ps_router = ns.network.NodeContainer()
            ps_router.Add(ps_clients_nodes_list[i].Get(0))
            ps_router.Add(nodes.Get(sn))
            ps_router_dev = p2p.Install(ps_router)
            ipv4_n.SetBase(ns.network.Ipv4Address("172.18.%d.0" % (i + 1)), ns.network.Ipv4Mask("255.255.255.0"))
            ipv4_n.Assign(ps_router_dev)

            # set up links between pses and clients, csma
            ps_clients_nodes = ps_clients_nodes_list[i]
            ps_clients_dev = csma.Install(ps_clients_nodes)
            ipv4_n.SetBase(ns.network.Ipv4Address("192.168.%d.0" % (i + 1)), ns.network.Ipv4Mask("255.255.255.0"))
            ipv4_n.Assign(ps_clients_dev)

        if verbose:
            print("Initialize Global Routing.")
        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

        return nodes, ps_clients_nodes_list, p2p

    def _allocate_node_position(self, coord_array=None, verbose=False):
        """Randomly allocate node positions or according to corrd_array"""

        if verbose:
            print("Allocate Positions to Nodes.")

        if coord_array is None:
            if verbose:
                print("Allocate Positions to Nodes.")
            mobility_n = ns.mobility.MobilityHelper()
            mobility_n.SetPositionAllocator("ns3::RandomDiscPositionAllocator",
                                            "X", ns.core.StringValue("100.0"),
                                            "Y", ns.core.StringValue("100.0"),
                                            "Rho", ns.core.StringValue("ns3::UniformRandomVariable[Min=0|Max=30]"))
            mobility_n.SetMobilityModel("ns3::ConstantPositionMobilityModel")
            mobility_n.Install(self._nodes)
        else:
            mobility_n = ns.mobility.MobilityHelper()
            positionAlloc_n = ns.mobility.ListPositionAllocator()

            for m in range(len(coord_array)):  # TODO, location related with delay ?
                positionAlloc_n.Add(ns.core.Vector(coord_array[m][0], coord_array[m][1], 0))
                n0 = self._nodes.Get(m)
                nLoc = n0.GetObject(ns.mobility.MobilityModel.GetTypeId())
                if nLoc is None:
                    nLoc = ns.mobility.ConstantPositionMobilityModel()
                    n0.AggregateObject(nLoc)
                # y-coordinates are negated for correct display in NetAnim
                # NetAnim's (0,0) reference coordinates are located on upper left corner
                # by negating the y coordinates, we declare the reference (0,0) coordinate
                # to the bottom left corner
                nVec = ns.core.Vector(coord_array[m][0], -coord_array[m][1], 0)
                nLoc.SetPosition(nVec)

            mobility_n.SetPositionAllocator(positionAlloc_n)
            mobility_n.Install(self._nodes)

        if verbose:
            for i in range(self._nodes.GetN()):
                position = self._nodes.Get(i).GetObject(ns.mobility.MobilityModel.GetTypeId())
                pos = position.GetPosition()
                print("Node %d: x=%d, y=%d" % (i, pos.x, pos.y))

    def _get_ps_nodes(self):
        return [self._ps_clients_nodes_list[i].Get(0) for i in range(self.ps_num)]
        # return [self._nodes.Get(sn) for sn in self._ps_sn_list]

    def _get_ps_node_by_sn(self, sn):
        return self._ps_clients_nodes_list[self._ps_sn_list.index(sn)].Get(0)

    def fl_step(self, start_time=0, stop_time=None, protocol='tcp', verbose=False):
        if verbose:
            print("Start simulating one round of FL.")

        assert protocol in ['tcp', 'udp']
        self.time_consumed_one_step = start_time

        def rcv_packet(socket):
            src = ns.network.Address()
            while True:
                packet = socket.RecvFrom(1024, 0, src)
                if packet is None or packet.GetSize() <= 0:
                    break
                rcv_time = ns.core.Simulator.Now().GetSeconds()
                if rcv_time > self.time_consumed_one_step:
                    self.time_consumed_one_step = rcv_time
                if verbose and ns.network.InetSocketAddress.IsMatchingType(src):
                    address = ns.network.InetSocketAddress.ConvertFrom(src)
                    print("At time %.6f packet sink received %d bytes from %s port %d" %
                          (rcv_time, packet.GetSize(), address.GetIpv4(), address.GetPort()))

        def accept_callback(a, b):
            return True

        def new_connection(socket, address):
            socket.SetRecvCallback(rcv_packet)

        def normal_close(socket):
            print("normal close")

        def error_close(socket):
            print("error close")

        port = 99
        for node in self._get_ps_nodes():
            sink_socket = ns.network.Socket.CreateSocket(node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(protocol.capitalize())))
            if protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
                # sink_socket.SetCloseCallbacks(normal_close, error_close)
            else:
                sink_socket.SetRecvCallback(rcv_packet)
            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

        # PSes aggregate models
        for i in range(self.ps_num):

            ps_node = self._ps_clients_nodes_list[i].Get(0)
            ipv4 = ps_node.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_int_addr = ipv4.GetAddress(1, 0)
            ip_addr = ipv4_int_addr.GetLocal()

            for j in range(1, self._ps_clients_nodes_list[i].GetN()):

                client_node = self._ps_clients_nodes_list[i].Get(j)

                sender = ns.applications.BulkSendHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                        ns.network.Address(
                                                            ns.network.InetSocketAddress(ip_addr, port)))
                sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size))
                apps_sender = sender.Install(client_node)
                apps_sender.Start(ns.core.Seconds(start_time))
                if stop_time is not None:
                    apps_sender.Stop(ns.core.Seconds(stop_time))

        ns.core.Simulator.Run()
        # ns.core.Simulator.Destroy()

        return self.time_consumed_one_step - start_time

    def pfl_step(self, times=10, start_time=0, stop_time=None, protocol='tcp', verbose=False):
        assert protocol in ['tcp', 'udp']
        self.time_consumed_one_step = start_time

        if verbose:
            # ns.core.LogComponentEnable("BulkSend", ns.core.LOG_LEVEL_INFO)
            ns.core.LogComponentEnable("PacketSink", ns.core.LOG_LEVEL_INFO)
        else:
            # ns.core.LogComponentEnable("BulkSend", ns.core.LOG_ERROR)
            ns.core.LogComponentEnable("PacketSink", ns.core.LOG_ERROR)

        m_totalRx = 0
        def rcv_packet(socket):
            src = ns.network.Address()
            while True:
                packet = socket.RecvFrom(1024, 0, src)
                if packet is None or packet.GetSize() <= 0:
                    break

                self.m_totalRx += packet.GetSize()

                rcv_time = ns.core.Simulator.Now().GetSeconds()
                if rcv_time > self.time_consumed_one_step:
                    self.time_consumed_one_step = rcv_time
                if verbose and ns.network.InetSocketAddress.IsMatchingType(src):
                    address = ns.network.InetSocketAddress.ConvertFrom(src)
                    print("At time %.6f packet sink received %d bytes from %s port %d" %
                          (rcv_time, packet.GetSize(), address.GetIpv4(), address.GetPort()))

                if self.m_totalRx == self.model_size:
                    print("received %d packets" % self.m_totalRx)

                    for x, y in self._target_overlay_graph.edges():
                        if x == y:
                            continue
                        # for src, dst in [(x, y), (y, x)]:
                        for _src, _dst in [(x, y)]:
                            if verbose:
                                print("Flow: %d -> %d" % (_src, _dst))

                            ipv4 = self._get_ps_node_by_sn(_dst).GetObject(ns.internet.Ipv4.GetTypeId())
                            ipv4_int_addr = ipv4.GetAddress(1, 0)
                            ip_addr = ipv4_int_addr.GetLocal()

                            sender = ns.applications.BulkSendHelper(
                                "ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                ns.network.Address(ns.network.InetSocketAddress(ip_addr, port)))

                            sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size))
                            apps_sender = sender.Install(self._nodes.Get(_src))
                            # apps_sender.Start()



        def accept_callback(a, b):
            return True

        def new_connection(socket, address):
            socket.SetRecvCallback(rcv_packet)

        def normal_close(socket):
            print("normal close")

        def error_close(socket):
            print("error close")

        # create sink application for PSes
        port = 9

        for node in self._get_ps_nodes():
            sink_socket = ns.network.Socket.CreateSocket(node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(protocol.capitalize())))
            if protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
                sink_socket.SetCloseCallbacks(normal_close, error_close)
            else:
                sink_socket.SetRecvCallback(rcv_packet)
            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

            # sink_helper = ns.applications.PacketSinkHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
            #                                                ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(),
            #                                                                             port))
            # sink_app = sink_helper.Install(node)
            # # packet_sink = sink_app.Get(0).GetObject(ns.applications.PacketSink.GetTypeId())
            # # packet_sink.TraceConnectWithoutContext("Rx", MakeCallBack)
            #
            # sink_app.Start(ns.core.Seconds(start_time))
            # sink_app.Stop(ns.core.Seconds(stop_time))

        # PSes mix models
        for x, y in self._target_overlay_graph.edges():
            if x == y:
                continue
            # for src, dst in [(x, y), (y, x)]:
            for src, dst in [(x, y)]:
                if verbose:
                    print("Flow: %d -> %d" % (src, dst))

                ipv4 = self._get_ps_node_by_sn(dst).GetObject(ns.internet.Ipv4.GetTypeId())
                ipv4_int_addr = ipv4.GetAddress(1, 0)
                ip_addr = ipv4_int_addr.GetLocal()

                sender = ns.applications.BulkSendHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                        ns.network.Address(ns.network.InetSocketAddress(ip_addr, port)))

                sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size))
                apps_sender = sender.Install(self._nodes.Get(src))
                apps_sender.Start(ns.core.Seconds(start_time))
                if stop_time is not None:
                    apps_sender.Stop(ns.core.Seconds(stop_time))

        if stop_time is not None:
            ns.core.Simulator.Stop(ns.core.Seconds(stop_time))
        ns.core.Simulator.Run()

        return self.time_consumed_one_step - start_time

    def ring_based_all_reduced(self, start_time=0, stop_time=None, protocol='tcp', verbose=False):

        # create sink application for PSes
        port = 9

        for node in self._get_ps_nodes():
            sink_socket = ns.network.Socket.CreateSocket(node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                           format(
                protocol.capitalize())))
            if protocol == 'tcp':
                sink_socket.SetAcceptCallback(accept_callback, new_connection)
            else:
                sink_socket.SetRecvCallback(rcv_packet)
            socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
            sink_socket.Bind(socket_address)
            sink_socket.Listen()

        # ring-based allreduce
        for x, y in self._target_overlay_graph.edges():
            if x == y:
                continue
            for src, dst in [(x, y), (y, x)]:
                # for src, dst in [(x, y)]:
                if verbose:
                    print("Flow: %d -> %d" % (src, dst))

                ipv4 = self._get_ps_node_by_sn(dst).GetObject(ns.internet.Ipv4.GetTypeId())
                ipv4_int_addr = ipv4.GetAddress(1, 0)
                ip_addr = ipv4_int_addr.GetLocal()

                sender = ns.applications.BulkSendHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                        ns.network.Address(
                                                            ns.network.InetSocketAddress(ip_addr, port)))

                sender.SetAttribute("MaxBytes", ns.core.UintegerValue(self.model_size/self.ps_num))
                apps_sender = sender.Install(self._nodes.Get(src))
                apps_sender.Start(ns.core.Seconds(start_time))
                if stop_time is not None:
                    apps_sender.Stop(ns.core.Seconds(stop_time))



    def all_reduced(self, start_time=0, stop_time=None, protocol='tcp', verbose=False):
        pass

    def plot_network(self, name='connectivity', node_label=False, figsize=(10,10)):
        fig, ax = plt.subplots(figsize=figsize)
        if name == 'underlay':
            graph = self._underlay_graph.copy()
        elif name == 'overlay':
            graph = self._target_overlay_graph.copy()
            # it is not necessary to display the loop-link
            for node in graph.nodes():
                graph.remove_edge(node, node)
        else:
            graph = self._target_connectivity_graph.copy()

        # plot graph
        pos = nx.spring_layout(graph)
        nx.draw_networkx(graph, width=2, alpha=0.8, with_labels=node_label,
                         style='--', edge_color='g', pos=pos, ax=ax)

        # plot edge labels
        if name == 'underlay':
            edge_labels = { (u, v): "%d ms" % (d['latency']*1000) for u, v, d in graph.edges(data=True)}  # ms
        else:
            edge_labels = {(u, v): "%d ms, %d Mbps" % (d['latency'] * 1000, d['availableBandwidth'] / 1e6)
                           for u, v, d in graph.edges(data=True)}  # ms, Mbps
        nx.draw_networkx_edge_labels(graph, edge_labels=edge_labels, pos=pos, ax=ax)

        # label target nodes
        if name == 'underlay':
            # edge_labels = { (u, v): "%d" % (d['distance']) for u, v, d in graph.edges(data=True)}  # kilometers
            nx.draw_networkx_nodes(self._target_overlay_graph, node_color='red', pos=pos, ax=ax)

        plt.title(name)
        plt.show()

    def plot_dis_bandwidth_parwise(self, figsize=(10, 8)):
        bandwidth_list = []
        for x, y, data in self._connectivity_graph.edges(data=True):
            bandwidth_list.append(data['availableBandwidth'])
        bandwidth_list.sort(reverse=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(1, len(bandwidth_list)+1), bandwidth_list)
        plt.show()

    def plot_flow_stat(self, figsize=(10,8)):
        self._monitor.CheckForLostPackets()
        classifier = self._flowmon_helper.GetClassifier()

        for flow_id, flow_stats in self._monitor.GetFlowStats():
            t = classifier.FindFlow(flow_id)
            proto = {6: 'TCP', 17: 'UDP'}[t.protocol]
            print("FlowID: %i (%s %s/%s --> %s/%i)" % \
                  (flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
            self._print_stats(sys.stdout, flow_stats)

        delays = []
        for flow_id, flow_stats in self._monitor.GetFlowStats():
            tupl = classifier.FindFlow(flow_id)
            if tupl.protocol == 17 and tupl.sourcePort == 698:
                continue
            delays.append(flow_stats.delaySum.GetSeconds() / flow_stats.rxPackets)
        plt.hist(delays, 20)
        plt.xlabel("Delay (s)")
        plt.ylabel("Number of Flows")
        plt.show()

    def _print_stats(self, os, st):
        print("  Tx Bytes: ", st.txBytes, file=os)
        print("  Rx Bytes: ", st.rxBytes, file=os)
        print("  Tx Packets: ", st.txPackets, file=os)
        print("  Rx Packets: ", st.rxPackets, file=os)
        print("  Lost Packets: ", st.lostPackets, file=os)
        if st.rxPackets > 0:
            print("  Mean{Delay}: ", (st.delaySum.GetSeconds() / st.rxPackets), file=os)
            print("  Mean{Jitter}: ", (st.jitterSum.GetSeconds() / (st.rxPackets - 1)), file=os)
            print("  Mean{Hop Count}: ", float(st.timesForwarded) / st.rxPackets + 1, file=os)

        print("Delay Histogram", file=os)
        for i in range(st.delayHistogram.GetNBins()):
            print(" ", i, "(", st.delayHistogram.GetBinStart(i), "-", \
                  st.delayHistogram.GetBinEnd(i), "): ", st.delayHistogram.GetBinCount(i), file=os)
        print("Jitter Histogram", file=os)
        for i in range(st.jitterHistogram.GetNBins()):
            print(" ", i, "(", st.jitterHistogram.GetBinStart(i), "-", \
                  st.jitterHistogram.GetBinEnd(i), "): ", st.jitterHistogram.GetBinCount(i), file=os)
        print("PacketSize Histogram", file=os)
        for i in range(st.packetSizeHistogram.GetNBins()):
            print(" ", i, "(", st.packetSizeHistogram.GetBinStart(i), "-", \
                  st.packetSizeHistogram.GetBinEnd(i), "): ", st.packetSizeHistogram)



if __name__ == "__main__":
    np.random.seed(123)
    client_group_list = [[1, 4, 9, 10, 13, 17, 19, 20, 52, 54, 60, 61, 64, 66, 68, 71, 76, 82, 89, 98],
                         [6, 7, 31, 33, 34, 35, 36, 39, 40, 43, 44, 55, 59, 62, 69, 70, 78, 92, 93, 99],
                         # [11, 14, 21, 25, 26, 30, 46, 49, 53, 58, 65, 72, 74, 79, 81, 85, 87, 90, 91, 96],
                         # [0, 8, 16, 22, 27, 29, 32, 41, 45, 47, 48, 57, 67, 73, 77, 83, 84, 86, 88, 97],
                         # [2, 3, 5, 12, 15, 18, 23, 24, 28, 37, 38, 42, 50, 51, 56, 63, 75, 80, 94, 95],
                         ]

    # network = Network(client_group_list=client_group_list, underlay_name='gaia',
    #                   parent_dir='.', model_size=1e7, link_capacity=1e9, verbose=False)
    network = Network(client_group_list=client_group_list, underlay_name='two', parent_dir='.',
                      link_capacity=1e6, node_capacity=1e6,
                      wan_latency=5e-17, lan_latency=5e-17,
                      model_size=1024, verbose=False)
    network.prepare(topo_name='ring', verbose=False)
    network.plot_network('underlay', node_label=True, figsize=(20, 20))
    network.plot_network('overlay', node_label=True, figsize=(4, 4))
    # network.plot_network(node_label=True, figsize=(6, 6))
    # network.plot_dis_bandwidth_parwise()

    # fl_start = 0
    # fl_stop = 10000
    # fl_time_consuming = network.fl_step(protocol='tcp', verbose=False, start_time=fl_start, stop_time=fl_stop)
    # print("fl:%f ms" % (fl_time_consuming * 1000))

    # pfl_start = 0
    # pfl_stop = 10000
    # pfl_time_consuming = network.pfl_step(protocol='tcp', verbose=False, start_time=pfl_start, stop_time=pfl_stop)
    # pfl_time_consuming -= fl_stop

    pfl_start = 0
    pfl_stop = 10000
    pfl_time_consuming = network.pfl_step(protocol='tcp', verbose=True, start_time=pfl_start, stop_time=pfl_stop)

    print("pfl:%f ms" % (pfl_time_consuming*1000))

    # network.plot_flow_stat()
    # [ebone, 8e6 bytes model size, 1e9 bps link capacity, 1e8 node_capacity]
    # pfl: (ring) 8971.399908 ms (1412.211356 ms, low wan delay),
    #      (complete) 10738.365724 ms (2817.959012 ms, low wan delay)
