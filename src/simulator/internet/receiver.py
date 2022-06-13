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
        self.message_queue = []

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

    def get_id(self):
        return self._id

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

            self.m_currentRxSize += packet.GetSize()
            rcv_time = ns.core.Simulator.Now().GetSeconds()
            self._current_time = max(rcv_time, self._current_time)

            if self.verbose:
                print("+ At time %.6f packet sink %d (receiver %d) received %d (%d) bytes from %s port %d" %
                      (rcv_time, self._communicator.get_id(), self._id, packet.GetSize(), self.m_currentRxSize,
                       address.GetIpv4(), address.GetPort()))

            # receives enough data in this phase and generate data for downstream node
            # note communicator may be offline at this moment
            if self.m_currentRxSize >= self.m_phaseRxSize * (self._current_phase + 1):
                self.complete_one_phase(address)

    def complete_one_phase(self, address):
        self._current_phase += 1
        self.message_queue.append(self._current_phase-1)

        if self.verbose:
            print("+ In %d-th phase packet sink %d (receiver %d) received %d total bytes" %
                  (self._current_phase - 1, self._communicator.get_id(), self._id, self.m_currentRxSize))

        # finish all phases
        if self._current_phase >= self.m_phases:
            if self.verbose:
                print("[Reception Finished] At time %.6f packet sink %d (receiver %d) received %d total bytes from %s port %d" %
                    (self._current_time, self._communicator.get_id(), self._id, self.m_currentRxSize, address.GetIpv4(),
                     address.GetPort()))
            self._is_finished = True
            self.StopApplication()
            return

        self._communicator.update_phase()
        # self._communicator.send_message()

    def fast_forward(self):
        if self.verbose:
            print("Fake: node %d received data from node %d" % (self._communicator.get_id(), self._id))
        self.m_currentRxSize = self.m_phaseRxSize * (self._current_phase + 1)
        ignoring_phase = self._current_phase
        self._current_phase += 1
        if self._current_phase >= self.m_phases:
            self._is_finished = True
            self.StopApplication()
            return ignoring_phase
        self._communicator.update_phase()
        return ignoring_phase

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
        self.message_queue = []
        if ns.network.InetSocketAddress.IsMatchingType(self.m_peer):
            self.m_socket.Bind(self.m_peer)
        else:
            self.m_socket.Bind6(self.m_peer)
        self.m_socket.Listen()

    def StopApplication(self):
        if self.m_socket:
            self.m_socket.Close()

    def get_current_time(self):
        return self._current_time

    def get_current_phase(self):
        return self._current_phase

    def is_finished(self):
        return self._is_finished