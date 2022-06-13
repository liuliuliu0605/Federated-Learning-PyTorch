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
import numpy as np

from matplotlib import pyplot as plt

# import ns.assert
# import ns.global_route_manager
# import ns.ipv4_global_routing_helper


#  source /data/magnolia/venv_set/torch/bin/activate
# cd /data/magnolia/ns-3-allinone/bake/source/ns-3.35 && ./waf shell
# cd /data/magnolia/Federated-Learning-PyTorch/src

# write the following environment variables into ~/.bashrc
# export PYTHONPATH=$PYTHONPATH:/data/magnolia/ns-3-allinone/bake/source/ns-3.35/build/bindings/python:/data/magnolia/ns-3-allinone/bake/source/ns-3.35/src/visualizer:/data/magnolia/ns-3-allinone/bake/source/pybindgen
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/magnolia/ns-3-allinone/bake/source/ns-3.35/build/lib

ns.core.LogComponentEnable("OnOffApplication", ns.core.LOG_LEVEL_INFO)
ns.core.LogComponentEnable("PacketSink",  ns.core.LOG_LEVEL_INFO)

verbose = False
# SimTime = 30.00
SinkStartTime = 0
SinkStopTime = 10
AppStartTime = 0
AppStopTime = 10
SimTime = 10
# # SinkStartTime = 1.0001
# # SinkStopTime = 20.90001
# AppStartTime = 0
# AppStopTime = 30

# CBR traffic: when an application is started, the first packet transmission occurs
# after a delay equal to (packet size/bit rate)
AppPacketSize = "1000"  # bytes
AppPacketRate = "1Gbps"
ns.core.Config.SetDefault("ns3::OnOffApplication::PacketSize", ns.core.StringValue(AppPacketSize))
ns.core.Config.SetDefault("ns3::OnOffApplication::MaxBytes", ns.core.StringValue(AppPacketSize))
ns.core.Config.SetDefault("ns3::OnOffApplication::DataRate", ns.core.StringValue(AppPacketRate))
# of dropped packets, default value:100
# ns.core.Config.SetDefault("ns3::DropTailQueue::MaxPackets", ns.core.UintegerValue(1000))
LinkRate = ns.core.StringValue("1Mbps")
LinkDelay = ns.core.StringValue("0ms")

tr_name = "n-node-ppp.tr"
pcap_name = "n-node-ppp"
flow_name = "n-node-ppp.xml"
anim_name = "n-node-ppp.anim.xml"

Adj_Matrix = [
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

coord_array = [[57.65, 47.6], [57.55, 47.25], [58.7, 38.59], [58.7, 37.97], [58, 37.35], [61.6, 34.1], [62.1, 33.8],
               [75.22, 41.13],
               [75, 39.7], [83.28, 32.95], [83.25, 32.78], [85.45, 39.12], [85.62, 38.91], [92.35, 41.85],
               [93.2, 39.83], [95.58, 33.75],
               [98.62, 28.54], [100.83, 35.48], [102.51, 39.04], [102.67, 38.95], [103, 38.9], [103.3, 39.23],
               [104.95, 39.95],
               [105.95, 40.13], [106, 40.71], [107.41, 42.1], [108.94, 42.36]]

# Adj_Matrix = [[0,1,0],[1,0,0],[0,1,0]]
# coord_array = [[0, 0], [0, 0],[0,0]]
Adj_Matrix = [[0, 1], [0, 1]]
coord_array = [[0, 0], [0, 0]]

n_nodes = len(Adj_Matrix)
matrixDimension = len(Adj_Matrix)
assert n_nodes == matrixDimension


# ---------- Network Setup ------------------------------------------------
if verbose:
    print("Create Nodes.")
nodes = ns.network.NodeContainer()
nodes.Create(n_nodes)

if verbose:
    print("Create P2P Link Attributes.")
p2p = ns.point_to_point.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", LinkRate)
p2p.SetChannelAttribute("Delay", LinkDelay)

if verbose:
    print("Install Internet Stack to Nodes.")
internet = ns.internet.InternetStackHelper()
internet.Install(nodes)

if verbose:
    print("Assign Addresses to Nodes.")
ipv4_n = ns.internet.Ipv4AddressHelper()
ipv4_n.SetBase(ns.network.Ipv4Address("10.0.0.0"), ns.network.Ipv4Mask("255.255.255.252"));

if verbose:
    print("Create Links Between Nodes.")
linkCount = 0

delay_list = ["0ms", "10ms", "20ms", "30ms"]
for i in range(len(Adj_Matrix)):
    for j in range(len(Adj_Matrix[i])):
        if Adj_Matrix[i][j] == 1:
            print(i,j)
            n_links = ns.network.NodeContainer()
            n_links.Add(nodes.Get(i))
            n_links.Add(nodes.Get(j))

            # LinkDelay = ns.core.StringValue(delay_list[1])
            # p2p.SetChannelAttribute("Delay", LinkDelay)
            n_devs = p2p.Install(n_links)

            # print(n_devs.Get(0).GetChannel().GetId()) #.GetObject(ns.mobility.MobilityModel.GetTypeId())
            # x = ns.core.AttributeValue()

            # n_devs.Get(0).SetMtu(1500)
            # n_devs.Get(1).SetMtu(1500)
            ipv4_n.Assign(n_devs)
            ipv4_n.NewNetwork()
            linkCount += 1
            if verbose:
                print("matrix element [", i, "][", j, "] is 1")
        else:
            if verbose:
                print("matrix element [", i, "][", j, "] is 0")

if verbose:
    print("Number of links in the adjacency matrix is: ", linkCount)
    print("Number of all nodes is: ", nodes.GetN())
    print("Initialize Global Routing.")
ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

# ---------- End of Network Set-up ----------------------------------------


# ---------- Allocate Node Positions --------------------------------------
if verbose:
    print("Allocate Positions to Nodes.");
mobility_n = ns.mobility.MobilityHelper()
positionAlloc_n = ns.mobility.ListPositionAllocator()

for m in range(len(coord_array)): #TODO, location related with delay ?
    positionAlloc_n.Add(ns.core.Vector(coord_array[m][0], coord_array[m][1], 0))
    n0 = nodes.Get(m)
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
mobility_n.Install(nodes)

# ---------- End of Allocate Node Positions -------------------------------


# ---------- Create n*(n-1) CBR Flows -------------------------------------
if verbose:
    print("Setup Packet Sinks.");
port = 9


def rcv_packet(socket):
    print("received", ns.core.Simulator.Now().GetSeconds())


def succeeded(a):
    # print("Connected")
    pass


def not_succeeded(a):
    print("ERROR: not connected")


def accept_callback(a, b):
    return True


def new_connection(socket, address):
    socket.SetRecvCallback(rcv_packet)

for i in range(n_nodes):
    # sink = ns.applications.PacketSinkHelper("ns3::TcpSocketFactory",
    #                                         ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port))
    # apps_sink = sink.Install(nodes.Get(i))  # sink is installed on all nodes
    # apps_sink.Start(ns.core.Seconds(SinkStartTime))
    # apps_sink.Stop(ns.core.Seconds(SinkStopTime))

    socket = ns.network.Socket.CreateSocket(nodes.Get(i), ns.core.TypeId.LookupByName("ns3::TcpSocketFactory"))
    # socket = ns.network.Socket.CreateSocket(nodes.Get(i), ns.core.TypeId.LookupByName("ns3::UdpSocketFactory"))
    socket.SetRecvCallback(rcv_packet)
    socket.SetConnectCallback(succeeded, not_succeeded)
    socket.SetAcceptCallback(accept_callback, new_connection)

    socket_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)
    socket.Bind(socket_address)
    socket.Listen()


if verbose:
    print("Setup CBR Traffic Sources.");

for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            # We needed to generate a random number (rn) to be used to eliminate
            # the artificial congestion caused by sending the packets at the
            # same time. This rn is added to AppStartTime to have the sources
            # start at different time, however they will still send at the same rate.
            x = ns.core.UniformRandomVariable()
            x.SetAttribute ("Min", ns.core.DoubleValue(0))
            x.SetAttribute ("Max", ns.core.DoubleValue(1))
            rn = x.GetValue()
            n = nodes.Get(j)
            ipv4 = n.GetObject(ns.internet.Ipv4.GetTypeId())
            ipv4_int_addr = ipv4.GetAddress(1, 0)
            ip_addr = ipv4_int_addr.GetLocal()
            onoff = ns.applications.OnOffHelper("ns3::TcpSocketFactory",
            # onoff = ns.applications.OnOffHelper("ns3::UdpSocketFactory",
                                                ns.network.Address(ns.network.InetSocketAddress(ip_addr, port)))  # traffic flows from node[i] to node[j]
            onoff.SetConstantRate(ns.network.DataRate(AppPacketRate))
            # onoff.SetConstantRate(ns.network.DataRate(AppPacketRate), int(AppPacketSize))
            onoff.SetAttribute ("MaxBytes", ns.core.UintegerValue(int(AppPacketSize)))

            # onoff.SetAttribute("OnTime", ns.core.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
            # onoff.SetAttribute("OffTime", ns.core.StringValue("ns3::ConstantRandomVariable[Constant=0]"))




            apps = onoff.Install(nodes.Get(i)) # traffic sources are installed on all nodes
            # apps.Start(ns.core.Seconds(AppStartTime + rn))
            apps.Start(ns.core.Seconds(AppStartTime))
            apps.Stop(ns.core.Seconds(AppStopTime))

# ---------- End of Create n*(n-1) CBR Flows ------------------------------


# ---------- Simulation Monitoring ----------------------------------------
if verbose:
    print("Configure Tracing.")
ascii = ns.network.AsciiTraceHelper()
p2p.EnableAsciiAll(ascii.CreateFileStream(tr_name))
# p2p.EnablePcapAll(pcap_name)

# TypeError: descriptor 'InstallAll' of 'flow_monitor.FlowMonitorHelper' object needs an argument
# flowmon = ns.flow_monitor.FlowMonitorHelper.InstallAll()

# Configure animator with default settings
anim = ns.netanim.AnimationInterface(anim_name)

# ns.core.Config.Connect("/NodeList/*/DeviceList/*/Tx", ns.core.MakeBoundCallback(my_callback))
# ns.core.Config.Connect("/NodeList/*/ApplicationList/0/$ns3::PacketSink/Rx", MakeCallback(&Rx_Trace))


flowmon_helper = ns.flow_monitor.FlowMonitorHelper()
#flowmon_helper.SetMonitorAttribute("StartTime", ns.core.TimeValue(ns.core.Seconds(31)))
monitor = flowmon_helper.InstallAll()
monitor = flowmon_helper.GetMonitor()
monitor.SetAttribute("DelayBinWidth", ns.core.DoubleValue(0.001))
monitor.SetAttribute("JitterBinWidth", ns.core.DoubleValue(0.001))
monitor.SetAttribute("PacketSizeBinWidth", ns.core.DoubleValue(20))


if verbose:
    print("Run Simulation.")
ns.core.Simulator.Stop(ns.core.Seconds(SimTime))
ns.core.Simulator.Run()
# ns.core.Simulator.Stop(ns.core.Seconds(SimTime))
# ns.core.Simulator.Run()
# flowmon->SerializeToXmlFile (flow_name.c_str(), true, true);
# print(ns.core.Simulator.Now().GetSeconds())
ns.core.Simulator.Destroy()

def print_stats(os, st):
    print("  Tx Bytes: ", st.txBytes, file=os)
    print("  Rx Bytes: ", st.rxBytes, file=os)
    print("  Tx Packets: ", st.txPackets, file=os)
    print("  Rx Packets: ", st.rxPackets, file=os)
    print("  Lost Packets: ", st.lostPackets, file=os)
    if st.rxPackets > 0:
        print("  Mean{Delay}: ", (st.delaySum.GetSeconds() / st.rxPackets), file=os)
        print("  Mean{Jitter}: ", (st.jitterSum.GetSeconds() / (st.rxPackets - 1)), file=os)
        print("  Mean{Hop Count}: ", float(st.timesForwarded) / st.rxPackets + 1, file=os)

    # if True:
    #     print("Delay Histogram", file=os)
    #     for i in range(st.delayHistogram.GetNBins()):
    #         print(" ", i, "(", st.delayHistogram.GetBinStart(i), "-", \
    #               st.delayHistogram.GetBinEnd(i), "): ", st.delayHistogram.GetBinCount(i), file=os)
    #     print("Jitter Histogram", file=os)
    #     for i in range(st.jitterHistogram.GetNBins()):
    #         print(" ", i, "(", st.jitterHistogram.GetBinStart(i), "-", \
    #               st.jitterHistogram.GetBinEnd(i), "): ", st.jitterHistogram.GetBinCount(i), file=os)
    #     print("PacketSize Histogram", file=os)
    #     for i in range(st.packetSizeHistogram.GetNBins()):
    #         print(" ", i, "(", st.packetSizeHistogram.GetBinStart(i), "-", \
    #               st.packetSizeHistogram.GetBinEnd(i), "): ", st.packetSizeHistogram.GetBinCount(i), file=os)

    for reason, drops in enumerate(st.packetsDropped):
        print("  Packets dropped by reason %i: %i" % (reason, drops), file=os)
    # for reason, drops in enumerate(st.bytesDropped):
    #    print "Bytes dropped by reason %i: %i" % (reason, drops)


monitor.CheckForLostPackets()
classifier = flowmon_helper.GetClassifier()

for flow_id, flow_stats in monitor.GetFlowStats():
    t = classifier.FindFlow(flow_id)
    proto = {6: 'TCP', 17: 'UDP'} [t.protocol]
    print ("FlowID: %i (%s %s/%s --> %s/%i)" % \
        (flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
    # print_stats(sys.stdout, flow_stats)

delays = []
for flow_id, flow_stats in monitor.GetFlowStats():
    tupl = classifier.FindFlow(flow_id)
    if tupl.protocol == 17 and tupl.sourcePort == 698:
        continue
    delays.append(flow_stats.delaySum.GetSeconds() / flow_stats.rxPackets)
plt.hist(delays, 20)
plt.xlabel("Delay (s)")
plt.ylabel("Number of Flows")
plt.show()


# ---------- End of Simulation Monitoring ---------------------------------