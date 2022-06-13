import ns.applications
import ns.core
import ns.internet
import ns.network
import ns.point_to_point
import numpy as np


class Receiver(ns.network.Application):

    def __init__(self, communicator=None):
        super(Receiver, self).__init__()
        self.m_socket = 0
        self.m_peer = ns.network.Address()
        self.m_phaseRxSize = 0
        self.m_currentRxSize = 0
        self.m_phases = 1
        self.verbose = False

        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        self._id = -1
        self._communicator = communicator

    def __del__(self):
        self.m_socket = 0

    def GetTypeId(self):
        tid = ns.core.TypeId("Receiver").SetParent().SetGroupName("Tutorial").AddConstructor()
        return tid

    def get_current_phase(self):
        return self._current_phase

    def accept_callback(self, a, b):
        return True

    def normal_close(self, socket):
        print("normal close")

    def error_close(self, socket):
        print("error close")

    def new_connection(self, socket, address):
        socket.SetRecvCallback(self.rcv_packet)

    def rcv_packet(self, socket):
        src = ns.network.Address()

        while True:

            packet = socket.RecvFrom(self.m_phaseRxSize, 0, src)
            if packet is None or packet.GetSize() <= 0:
                break

            if ns.network.InetSocketAddress.IsMatchingType(src):
                address = ns.network.InetSocketAddress.ConvertFrom(src)

            if self._communicator.is_offline():
                for sender in self._communicator.get_upstream_app_sender_list():
                    addr = ns.network.Address()
                    sender.m_socket.GetSockName(addr)
                    if addr == src:
                        sender.m_currentTxSize -= packet.GetSize()
                        break
                if self.verbose:
                    print("* At time %.6f packet sink (%d) abandoned %d (%d) bytes from %s port %d" %
                          (ns.core.Simulator.Now().GetSeconds(), self._id, packet.GetSize(), self.m_currentRxSize,
                           address.GetIpv4(), address.GetPort()))
                break

            self.m_currentRxSize += packet.GetSize()
            rcv_time = ns.core.Simulator.Now().GetSeconds()
            self._current_time = max(rcv_time, self._current_time)

            if self.verbose:
                print("+ At time %.6f packet sink (%d) received %d (%d) bytes from %s port %d" %
                      (rcv_time, self._id, packet.GetSize(), self.m_currentRxSize, address.GetIpv4(), address.GetPort()))

            if self.m_currentRxSize >= self.m_phaseRxSize * (self._current_phase + 1):
                self._communicator.alter_state()

                self._current_phase += 1

                # while self.m_currentRxSize >= self.m_phaseRxSize * (self._current_phase + 1):
                #     self._current_phase += 1

                if self._current_phase >= self.m_phases:
                    if self.verbose:
                        print("[Reception Finished] At time %.6f packet sink (%d) received %d total bytes from %s port %d" %
                              (rcv_time, self._id, self.m_currentRxSize, address.GetIpv4(), address.GetPort()))
                    self._is_finished = True
                    self.StopApplication()
                else:
                    if self.verbose:
                        print("+ In %d-th phase packet sink (%d) received %d total bytes" %
                              (self._current_phase-1, self._id, self.m_currentRxSize))
                    if self._communicator.is_offline():
                        self.offline_operation()
                        tNext = ns.core.Time(ns.core.Seconds(self._communicator.get_offline_duration()))
                        ns.core.Simulator.Schedule(tNext, self.online_operation)
                    else:
                        self.start_local_senders()

    # def Schedule(self):
    #     tNext = ns.core.Time(ns.core.Seconds(self.m_packetSize * 8 / self.m_dataRate.GetBitRate()))
    #     # tNext =  ns.core.Time(ns.core.Seconds(0))
    #     self.m_sendEvent = ns.core.Simulator.Schedule(tNext, self.SendPacket)

    def offline_operation(self):
        # cannot send or receive data
        if self.verbose:
            print("# At time %.6f %d node is offline in %d-th phase" %
                  (ns.core.Simulator.Now().GetSeconds(), self._id, self._current_phase))
        self.deactivate_local_senders()
        self.deactivate_upstream_senders()

    def online_operation(self):
        if self.verbose:
            print("@ At time %.6f %d node is online in %d-th phase" %
                  (ns.core.Simulator.Now().GetSeconds(), self._id, self._current_phase))
        self._communicator.make_online()
        self.activate_local_senders()
        self.activate_upstream_senders()
        for sender in self._communicator.get_app_sender_list():
            sender.ScheduleTx()
        for sender in self._communicator.get_upstream_app_sender_list():
            sender.ScheduleTx()

    def deactivate_upstream_senders(self):
        for sender in self._communicator.get_upstream_app_sender_list():
            sender.deactivate()

    def activate_upstream_senders(self):
        for sender in self._communicator.get_upstream_app_sender_list():
            sender.activate()

    def deactivate_local_senders(self):
        for sender in self._communicator.get_app_sender_list():
            sender.deactivate()

    def activate_local_senders(self):
        for sender in self._communicator.get_app_sender_list():
            sender.activate()

    def start_local_senders(self):
        for sender in self._communicator.get_app_sender_list():
            sender.ScheduleTx()

    def Setup(self, socket, address, phaseTxSize, phases=1, protocol='tcp', verbose=False, id=-1):
        self.m_socket = socket
        self.m_peer = address
        self.m_phaseRxSize = phaseTxSize
        self.m_phases = phases
        self.verbose = verbose
        self._id = id

        if protocol == 'tcp':
            self.m_socket.SetAcceptCallback(self.accept_callback, self.new_connection)
            # sink_socket.SetCloseCallbacks(normal_close, error_close)
        else:
            self.m_socket.SetRecvCallback(self.rcv_packet)

    def StartApplication(self):
        self.m_currentRxSize = 0
        self._current_time = 0
        self._current_phase = 0
        self._is_finished = False
        if ns.network.InetSocketAddress.IsMatchingType(self.m_peer):
            self.m_socket.Bind(self.m_peer)
        else:
            self.m_socket.Bind6(self.m_peer)
        self.m_socket.Listen()

    def StopApplication(self):
        if self.m_socket:
            self.m_socket.Close()
        self._app_local_sender_list = []
        self._app_upstream_sender_list = []

    def get_current_time(self):
        return self._current_time

    def is_finished(self):
        return self._is_finished