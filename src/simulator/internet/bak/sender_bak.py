import ns.applications
import ns.core
import ns.internet
import ns.network
import ns.point_to_point


class Sender(ns.network.Application):

    def __init__(self, communicator=None):
        super(Sender, self).__init__()
        self.m_socket = 0
        self.m_peer = ns.network.Address()
        self.m_phaseTxSize = 0
        self.m_phases = 1
        self.m_packetSize = 0
        self.m_dataRate = 0
        self.m_sendEvent = ns.core.EventId()
        self.m_running = False
        self.m_currentTxSize = 0
        self.verbose = False

        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        self._id = -1
        self._is_active = True
        self._communicator = communicator
        # self._address_list = []

    def __del__(self):
        self.m_socket = 0
        # self._address_list = []

    @staticmethod
    def GetTypeId(self):
        tid = ns.core.TypeId("Sender").SetParent().SetGroupName("Tutorial").AddConstructor()
        return tid

    def Setup(self, socket, address, packetSize, phaseTxSize, dataRate, phases=1, verbose=False, id=-1):
        self.m_socket = socket
        self.m_peer = address
        self.m_packetSize = packetSize
        self.m_phaseTxSize = phaseTxSize
        self.m_phases = phases
        self.m_dataRate = dataRate
        self.m_currentTxSize = 0
        self.verbose = verbose
        self._id = id

    def StartApplication(self):
        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        self._is_active = True
        if ns.network.InetSocketAddress.IsMatchingType(self.m_peer):
            self.m_socket.Bind()
        else:
            self.m_socket.Bind6()
        self.m_socket.Connect(self.m_peer)
        self.SendPacket()

    def StopApplication(self):
        if self.m_sendEvent.IsRunning():
            ns.core.Simulator.Cancel(self.m_sendEvent)
        if self.m_socket:
            self.m_socket.Close()

    def ScheduleTx(self):
        tNext =  ns.core.Time(ns.core.Seconds(self.m_packetSize * 8 / self.m_dataRate.GetBitRate()))
        # tNext =  ns.core.Time(ns.core.Seconds(0))
        self.m_sendEvent = ns.core.Simulator.Schedule(tNext, self.SendPacket)

    def deactivate(self):
        if self.verbose:
            print("%d sender is deactivated" % self._id)
        self._is_active = False

    def activate(self):
        if self.verbose:
            print("%d sender is activated" % self._id)
        self._is_active = True

    def SendPacket(self):
        if self.m_socket.GetTxAvailable() <= 0 or not self._is_active:
            return

        left = self.m_phaseTxSize * (self._current_phase + 1) - self.m_currentTxSize
        data_offset = self.m_currentTxSize % self.m_packetSize
        to_write = self.m_packetSize - data_offset
        to_write = min(to_write, left)
        to_write = min(to_write, self.m_socket.GetTxAvailable())
        packet = ns.network.Packet(to_write)
        amount_sent = self.m_socket.Send(packet, 0)

        if amount_sent <= 0:
            # print(to_write, self.m_packetSize, left, data_offset)
            print("Warning: no data transmission for packet source (%d)!" % self._id)
            return

        self._current_time = max(self._current_time, ns.core.Simulator.Now().GetSeconds())
        self.m_currentTxSize += amount_sent
        if self.verbose:
            print("- At time %.6f packet source (%d) sent %d (%d) bytes to %s port %d" %
                  (self._current_time, self._id,amount_sent, self.m_currentTxSize,
                   self.m_peer.GetIpv4(), self.m_peer.GetPort()))

        if self.m_currentTxSize < self.m_phaseTxSize * (self._current_phase + 1):
            self.ScheduleTx()
        else:
            self._current_phase += 1
            # while self.m_currentTxSize >= self.m_phaseTxSize * (self._current_phase + 1):
            #     self._current_phase += 1
            if self.verbose:
                print("- In %d-th phase packet source (%d) sent %d total bytes" %
                      (self._current_phase-1, self._id, self.m_currentTxSize))
            if self._current_phase >= self.m_phases:
                if self.verbose:
                    print("[Transmission Finished] At time %.6f packet source (%d) sent %d total bytes to %s port %d" %
                          (self._current_time, self._id, self.m_currentTxSize, self.m_peer.GetIpv4(), self.m_peer.GetPort()))
                self._is_finished = True
                self.StopApplication()
            elif self._current_phase <= self._communicator.get_app_receiver_list()[0].get_current_phase():
                # receiving phase is faster than sending phase, send buffer data
                self.ScheduleTx()
            elif len(self._communicator.get_app_receiver_list()) == 0:
                self.ScheduleTx()

    def get_current_time(self):
        return self._current_time

    def is_finished(self):
        return self._is_finished


if __name__ == "__main__":
    ns.core.LogComponentEnable("PacketSink", ns.core.LOG_LEVEL_INFO)

    protocol = 'tcp'
    port = 1080

    nodes = ns.network.NodeContainer()
    nodes.Create(2)

    pointToPoint = ns.point_to_point.PointToPointHelper()
    pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
    pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("0ms"))

    devices = pointToPoint.Install(nodes)

    stack = ns.internet.InternetStackHelper()
    stack.Install(nodes)

    address = ns.internet.Ipv4AddressHelper()
    address.SetBase(ns.network.Ipv4Address("10.1.1.0"),
                    ns.network.Ipv4Mask("255.255.255.0"))

    interfaces = address.Assign(devices)

    sink_helper = ns.applications.PacketSinkHelper("ns3::{:s}SocketFactory".format(protocol.capitalize()),
                                                   ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(),
                                                                                port))
    sink_app = sink_helper.Install(nodes.Get(1))
    sink_app.Start(ns.core.Seconds(0))
    sink_app.Stop(ns.core.Seconds(10))
    sinkAddress = ns.network.InetSocketAddress(interfaces.GetAddress(1), port)


    socket = ns.network.Socket.CreateSocket(nodes.Get(0), ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                      format(protocol.capitalize())))
    app = Sender()
    app.Setup(socket, sinkAddress, 536, 1000000, ns.network.DataRate("10Mbps"))
    nodes.Get(0).AddApplication(app)

    app.SetStartTime(ns.core.Seconds(0))
    app.SetStopTime(ns.core.Seconds(10))

    ns.core.Simulator.Run()
    ns.core.Simulator.Destroy()