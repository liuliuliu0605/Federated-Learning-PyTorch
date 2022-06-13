import ns.core
import numpy as np

from src.simulator.internet.sender import Sender
from src.simulator.internet.receiver import Receiver


class Communicator:

    def __init__(self, ns_node, id=-1, offline_params={}, protocol='tcp', verbose=False,
                 global_comm_matrix=None):
        self.ns_node = ns_node
        self.protocol = protocol
        self.verbose = verbose
        self.offline_params = offline_params
        self.global_comm_matrix = global_comm_matrix

        self._sink_socket_list = []
        self._source_socket_list = []
        self._app_sender_dict = {}
        self._app_receiver_dict = {}
        self._upstream_app_sender_dict = {}

        self._is_offline = False
        self._id = id
        self._next_phase = 1

    def __del__(self):
        self.reset()

    def get_id(self):
        return self._id

    def reset(self):
        for sink_socket in self._sink_socket_list:
            sink_socket.Close()
        self._sink_socket_list.clear()
        for source_socket in self._source_socket_list:
            source_socket.Close()
        self._source_socket_list.clear()
        self._app_sender_dict = {}
        self._app_receiver_dict = {}
        self._is_offline = False
        self._id = -1

    def get_offline_duration(self):
        return self.offline_params.get("duration", 0)

    def is_offline(self):
        return self._is_offline

    def get_app_receiver_dict(self):
        return self._app_receiver_dict

    def get_app_sender_dict(self):
        return self._app_sender_dict

    def get_current_time(self):
        current_time = 0
        for sender in self._app_sender_dict.values():
            current_time = max(current_time, sender.get_current_time())
        for receiver in self._app_receiver_dict.values():
            current_time = max(current_time, receiver.get_current_time())
        return current_time

    def is_finished(self):
        sender_states = [sender.is_finished() for sender in self._app_sender_dict.values()]
        receiver_states = [receiver.is_finished() for receiver in self._app_receiver_dict.values()]
        return np.all(sender_states) and np.all(receiver_states)

    def add_app_receiver(self, comm_id, phase_rx_size, phases, port, start_time=0, stop_time=None):
        sink_node = self.ns_node
        sink_socket = ns.network.Socket.CreateSocket(sink_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                            format(self.protocol.capitalize())))
        sink_address = ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port)

        app_receiver = Receiver(communicator=self)
        app_receiver.Setup(sink_socket, sink_address, phase_rx_size, phases=phases,
                           protocol=self.protocol, verbose=self.verbose, id=comm_id)
        sink_node.AddApplication(app_receiver)
        app_receiver.SetStartTime(ns.core.Seconds(start_time))
        if stop_time is not None:
            app_receiver.SetStopTime(ns.core.Seconds(stop_time))

        self._sink_socket_list.append(sink_socket)
        self._app_receiver_dict[comm_id] = app_receiver

        return app_receiver

    def add_app_sender(self, comm_id, dst_node, phase_rx_size, phases, port, packet_size, data_rate,
                       start_time=0, stop_time=None):
        src_node = self.ns_node
        ipv4 = dst_node.GetObject(ns.internet.Ipv4.GetTypeId())
        ipv4_int_addr = ipv4.GetAddress(1, 0)
        ip_addr = ipv4_int_addr.GetLocal()
        sink_address = ns.network.InetSocketAddress(ip_addr, port)
        source_socket = ns.network.Socket.CreateSocket(src_node, ns.core.TypeId.LookupByName("ns3::{:s}SocketFactory".
                                                                                             format(self.protocol.capitalize())))

        app_sender = Sender(communicator=self)
        app_sender.Setup(source_socket, sink_address, packet_size, phase_rx_size,
                          ns.network.DataRate("{:f}bps".format(data_rate)), phases=phases,
                          verbose=self.verbose, id=comm_id)
        src_node.AddApplication(app_sender)
        app_sender.SetStartTime(ns.core.Seconds(start_time))
        if stop_time is not None:
            app_sender.SetStopTime(ns.core.Seconds(stop_time))

        self._source_socket_list.append(source_socket)
        self._app_sender_dict[comm_id] = app_sender

        return app_sender

    def associate_upstream_app_sender(self, comm_id, sender):
        self._upstream_app_sender_dict[comm_id] = sender

    def offline_operation(self):
        if self._is_offline:
            return
        # communicator itself and its predecessors will stop sending data
        if self.verbose:
            print("# At time %.6f node %d is offline in %d-th phase" %
                  (ns.core.Simulator.Now().GetSeconds(), self._id, self._next_phase-1))
        self.deactivate_local_senders()
        self.deactivate_upstream_senders()
        self._is_offline = True

    def online_operation(self):
        if not self._is_offline:
            return
        # communicator itself and its predecessors will start sending data
        if self.verbose:
            print("@ At time %.6f node %d is online in %d-th phase" %
                  (ns.core.Simulator.Now().GetSeconds(), self._id, self._next_phase-1))
        self._is_offline = False
        self.activate_local_senders()
        self.activate_upstream_senders()
        # self.send_message()
        # self.inform_upstream_send_message()

    def switch_offline(self, current_phase):
        p = np.random.random()
        is_offline = False if self.offline_params.get('probability') is None or p >= self.offline_params['probability'] else True

        if is_offline:
            # get offline now and get online after some time
            self.offline_operation(current_phase)
            online_time = ns.core.Time(ns.core.Seconds(self.get_offline_duration()))
            ns.core.Simulator.Schedule(online_time, self.online_operation, current_phase)

            # UNBLOCKING offline nodes will not participate the updating of the current phase
            # for sender in self._app_sender_list:
            #     self.update_global_comm_matrix(current_phase, sender.get_id())

        # else:
            # self.online_operation(current_phase)
        return is_offline

    def update_phase(self):
        update = np.all([receiver.get_current_phase() >= self._next_phase for receiver
                         in self._app_receiver_dict.values()])
        if update:
            if self.verbose:
                print("Node %d entered %d-th phase" % (self._id, self._next_phase))
            self.generate_message()
            self.send_message()
            self._next_phase += 1

    def get_lagging_communicator_ids(self):
        lagging_list = []
        for comm_id in self._app_receiver_dict:
            if self._app_receiver_dict[comm_id].get_current_phase() < self._next_phase:
                lagging_list.append(comm_id)
        return lagging_list

    def generate_message(self, message=None):
        if self.verbose:
            print("Node %d generate new message" % self._id)
        message = self._next_phase if message is None else message
        for sender in self._app_sender_dict.values():
            sender.add_message(message)

    def send_message(self):
        if self.is_offline():
            return
        if self.verbose:
            print("Node %d send messages to neighbours" % self._id)
        for sender in self._app_sender_dict.values():
            sender.ScheduleTx()

    def inform_upstream_send_message(self):
        for sender in self._upstream_app_sender_dict.values():
            sender.get_communicator().send_message()

    def deactivate_upstream_senders(self):
        for sender in self._upstream_app_sender_dict.values():
            sender.deactivate()

    def activate_upstream_senders(self):
        for sender in self._upstream_app_sender_dict.values():
            sender.activate()

    def deactivate_local_senders(self):
        for sender in self._app_sender_dict.values():
            sender.deactivate()

    def activate_local_senders(self):
        for sender in self._app_sender_dict.values():
            sender.activate()

    # def abandon_data_from(self, comm):
    #     abandon_phase = -1
    #     receiver = self._app_receiver_dict.get(comm.get_id(), None)
    #     sender = comm.get_app_sender_dict().get(self._id, None)
    #     if receiver is not None and sender is not None:
    #         phase_a = sender.fast_forward()
    #         phase_b = receiver.fast_forward()
    #         assert phase_a == phase_b
    #         abandon_phase = phase_a
    #     return abandon_phase

    def abandon_data_from(self, comm, abandon_phase=-1):
        receiver = self._app_receiver_dict.get(comm.get_id(), None)
        sender = comm.get_app_sender_dict().get(self._id, None)

        if receiver is None or sender is None or len(sender.message_queue) == 0 or \
                sender.get_current_phase() != receiver.get_current_phase():
            return None

        if abandon_phase >= 0 and (sender.get_current_phase() != abandon_phase or
                                    receiver.get_current_phase() != abandon_phase):
            return None

        if self.verbose:
            print("Node %d started to abandoned data from node %d" %
                  (self._id, comm.get_id()))

        phase_a = sender.fast_forward()
        phase_b = receiver.fast_forward()
        # print(phase_a, phase_b)
        assert phase_a == phase_b
        return phase_a

    def update_global_comm_matrix(self, phase, sender_id):
        self.global_comm_matrix[phase][self._id, sender_id] += 1
