import ns.applications
import ns.core
import ns.internet
import ns.network
import ns.point_to_point

import threading


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
        self.message_queue = [0]
        self.verbose = False

        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        self._id = -1
        self._counts_of_deactivation = 0
        self._lock = threading.Lock()
        self._communicator = communicator

    def __del__(self):
        self.m_socket = 0

    @staticmethod
    def GetTypeId(self):
        tid = ns.core.TypeId("Sender").SetParent().SetGroupName("Tutorial").AddConstructor()
        return tid

    def get_id(self):
        return self._id

    def get_communicator(self):
        return self._communicator

    def Setup(self, socket, address, packetSize, phaseTxSize, dataRate, phases=1, verbose=False, id=-1):
        self.m_socket = socket
        self.m_peer = address
        self.m_packetSize = packetSize
        self.m_phaseTxSize = phaseTxSize
        self.m_phases = phases
        self.m_dataRate = dataRate
        self.verbose = verbose
        self._id = id

    def StartApplication(self):
        self.m_currentTxSize = 0
        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        self.message_queue = [0]
        if ns.network.InetSocketAddress.IsMatchingType(self.m_peer):
            self.m_socket.Bind()
        else:
            self.m_socket.Bind6()
        self.m_socket.Connect(self.m_peer)
        # self.SendPacket()

    def StopApplication(self):
        if self.m_sendEvent.IsRunning():
            ns.core.Simulator.Cancel(self.m_sendEvent)
        if self.m_socket:
            self.m_socket.Close()
        self._counts_of_deactivation = 0

    def deactivate(self):
        # self._lock.acquire()
        self._counts_of_deactivation += 1
        # self._lock.release()
        if self.verbose:
            print("Node %d (sender %d) is deactivated (semaphore=%d)"
                  % (self._communicator.get_id(), self._id, self._counts_of_deactivation))

    def activate(self):
        # self._lock.acquire()
        self._counts_of_deactivation -= 1
        # self._lock.release()
        if self.verbose:
            print("Node %d (sender %d) is activated (semaphore=%d)"
                  % (self._communicator.get_id(), self._id, self._counts_of_deactivation))

    def is_active(self):
        # self._lock.acquire()
        assert self._counts_of_deactivation >= 0
        state = self._counts_of_deactivation == 0
        # self._lock.release()
        return state

    def check_before_sending(self):
        ready = True
        warning_message = None
        if len(self.message_queue) == 0:
            ready = False
            warning_message = "Warning: node %d (sender %d) message queue was empty!" % \
                              (self._communicator.get_id(), self._id)
        elif self.m_sendEvent.IsRunning():
            ready = False
            warning_message = "Warning: there was already one message being sent!"
        elif self.m_socket.GetTxAvailable() <= 0:
            ready = False
            warning_message = "Warning: not available sending buffer!"
        elif not self.is_active():
            ready = False
            warning_message = "Warning: node %d (sender %d) was inactive!" % (self._communicator.get_id(), self._id)
        if self.verbose and warning_message is not None:
            print(warning_message)
        return ready

    def ScheduleTx(self, next_time=None):
        if not self.check_before_sending():
            return
        if next_time is not None:
            t_next = ns.core.Time(ns.core.Seconds(0))
        else:
            t_next = ns.core.Time(ns.core.Seconds(self.m_packetSize * 8 / self.m_dataRate.GetBitRate()))
        self.m_sendEvent = ns.core.Simulator.Schedule(t_next, self.SendPacket)

    def SendPacket(self):
        if not self.check_before_sending():
            return

        left = self.m_phaseTxSize * (self._current_phase + 1) - self.m_currentTxSize
        data_offset = self.m_currentTxSize % self.m_packetSize
        to_write = self.m_packetSize - data_offset
        to_write = min(to_write, left)
        to_write = min(to_write, self.m_socket.GetTxAvailable())
        packet = ns.network.Packet(to_write)
        amount_sent = self.m_socket.Send(packet, 0)

        if amount_sent <= 0:
            print(to_write, self.m_packetSize, left, data_offset)
            print("Warning: no data transmission for packet source %d (sender %d)!" %
                  (self._communicator.get_id(), self._id))
            return

        self._current_time = max(self._current_time, ns.core.Simulator.Now().GetSeconds())
        self.m_currentTxSize += amount_sent
        if self.verbose:
            print("- At time %.6f packet source %d (sender %d) sent %d (%d) bytes to %s port %d" %
                  (self._current_time, self._communicator.get_id(), self._id, amount_sent, self.m_currentTxSize,
                   self.m_peer.GetIpv4(), self.m_peer.GetPort()))

        if self.m_currentTxSize < self.m_phaseTxSize * (self._current_phase + 1):
            # it has not received enough data
            self.ScheduleTx()
        else:
            self.complete_one_phase()

            # # id = 0 means only one sender can trigger offline function
            # # because one communicator may contain several senders
            # if self._id == 0 and self._communicator.switch_offline(self._current_phase):
            #     return

    def complete_one_phase(self):
        self._current_phase += 1
        self.message_queue.pop(0)
        self.get_communicator().update_global_comm_matrix(self._current_phase - 1, self._id)
        if self.verbose:
            print("- In %d-th phase packet source %d (sender %d) sent %d total bytes" %
                  (self._current_phase - 1, self._communicator.get_id(), self._id, self.m_currentTxSize))

        if self._current_phase >= self.m_phases:
            if self.verbose:
                print("[Transmission Finished] At time %.6f packet source %d (sender %d) sent %d total bytes to %s port %d" %
                    (self._current_time, self._communicator.get_id(), self._id, self.m_currentTxSize,
                     self.m_peer.GetIpv4(), self.m_peer.GetPort()))
            self._is_finished = True
            self.StopApplication()
            return

        # check message queue to send the next message
        if len(self.message_queue) > 0:
            self.get_communicator().send_message()

    def add_message(self, msg=None):
        self.message_queue.append(msg)

    def fast_forward(self):
        if self.verbose:
            print("Fake: node %d sending data to node %d" % (self._communicator.get_id(), self._id))
        assert len(self.message_queue) > 0
        self._current_time = max(self._current_time, ns.core.Simulator.Now().GetSeconds())
        self.m_currentTxSize = self.m_phaseTxSize * (self._current_phase + 1)
        ignoring_phase = self._current_phase
        self._current_phase += 1
        self.message_queue.pop(0)
        if self._current_phase >= self.m_phases:
            self._is_finished = True
            self.StopApplication()

        return ignoring_phase

    def get_current_time(self):
        return self._current_time

    def get_current_phase(self):
        return self._current_phase

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