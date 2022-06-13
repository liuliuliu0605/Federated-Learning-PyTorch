import ns.core
import numpy as np

from src.simulator.internet.communicator import Communicator

CHECK_FREQUENCE = 0.01

class DecentralizedConsensus:

    def __init__(self, model_size, ps_node_container, matrix, data_rate=1e9, packet_size=536,
                 protocol='tcp', port=9, verbose=False, offline_params={'probability':0., 'duration': 0.},
                 max_block_duration=9999999999999):
        assert protocol in ['tcp', 'udp']
        assert ps_node_container.GetN() == len(matrix)

        self.ps_node_container = ps_node_container
        self.model_size = model_size
        self.matrix = matrix
        self.data_rate = data_rate
        self.packet_size = packet_size

        self.protocol = protocol
        self.port = port

        self.node_num = self.ps_node_container.GetN()
        self.verbose = verbose
        self.offline_params = offline_params

        self._time_consuming = 0
        self.communicator_list = None
        self.global_comm_matrix_list = None
        self.max_block_duration = max_block_duration

    def __del__(self):
        self.reset()

    def reset(self):
        self._time_consuming = 0
        for communicator in self.communicator_list:
            communicator.reset()

    def set_sink_nodes(self, start_time=0, stop_time=None, phases=1):
        for i in range(self.node_num):
            phase_rx_size = (np.count_nonzero(self.matrix[:, i] > 0) - 1) * self.model_size
            if phase_rx_size <= 0:
                continue
            self.communicator_list[i].add_app_receiver(phase_rx_size, phases, self.port, start_time, stop_time)

    def set_source_nodes(self, start_time=0, stop_time=None, phases=1):
        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.matrix[src, dst] > 0:

                    if self.verbose:
                        print("Mixing model: PS %d -> PS %d" % (src, dst))
                    dst_node = self.ps_node_container.Get(dst)
                    app_sender = self.communicator_list[src].add_app_sender(dst_node, dst, self.model_size, phases,
                                                                            self.port, self.packet_size, self.data_rate,
                                                                            start_time, stop_time)
                    self.communicator_list[dst].associate_upstream_app_sender(app_sender)

    def get_time_consuming(self):
        return self._time_consuming

    def is_finishied(self):
        communicator_states = [communicator.is_finished() for communicator in self.communicator_list]
        return np.all(communicator_states)

    def unblock_thread(self):
        for i, communicator_x in enumerate(self.communicator_list):

            if communicator_x.is_offline():

                for dst, sender in zip(communicator_x.get_sender_dst_list(), communicator_x.get_app_sender_list()):

                    # check whether communicator y is blocked by offline communicator x
                    communicator_y = self.communicator_list[dst]
                    # if not communicator_y.is_offline() and communicator_y.is_blocked_by(communicator_x):
                    if communicator_y.is_blocked_by(communicator_x):
                        if self.verbose:
                            print("%d was blocked by %d" % (communicator_y.get_id(), communicator_x.get_id()))
                        # unblock the online communicator y if exceeding maximum block duration
                        if ns.core.Simulator.Now().GetSeconds() - communicator_y.get_current_time() > self.max_block_duration:
                            if self.verbose:
                                print("%d started to unblocked %d" % (communicator_x.get_id(), communicator_y.get_id()))
                            ignoring_phase = communicator_x.unblock(communicator_y)
                        #     print("%d unblocked %d" % (communicator_x.get_id(), communicator_y.get_id()))
                            if self.verbose:
                                print("Node %d will not send data to node %d in %d-th phased"
                                      % (communicator_x.get_id(), communicator_y.get_id(), ignoring_phase))

                # communicator.unblock()
            # predecessor_list = [self.communicator_list[pre] for pre in self.matrix[:,j]]
            # for predecessor in predecessor_list:
            #     if predecessor.is_offline():
            #         print("Unblocking!")
            #         predecessor

        if not self.is_finishied():
            ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(CHECK_FREQUENCE)), self.unblock_thread)

    def run(self, start_time, stop_time, phases=1):
        # self.reset()
        self.global_comm_matrix_list = [np.zeros((self.node_num, self.node_num)) for _ in range(phases)]
        self.communicator_list = [Communicator(self.ps_node_container.Get(i), i, self.offline_params,
                                               self.protocol, self.verbose, self.global_comm_matrix_list)
                                  for i in range(self.node_num)]

        self.set_sink_nodes(start_time, stop_time, phases)
        self.set_source_nodes(start_time, stop_time, phases)

        for i in range(len(self.communicator_list)):
            # self.communicator_list[i].send_message()
            if not self.communicator_list[i].switch_offline(0):
                ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(0)), self.communicator_list[i].send_message)

        ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(CHECK_FREQUENCE)), self.unblock_thread)

        start_of_simulation = ns.core.Simulator.Now().GetSeconds() + start_time
        ns.core.Simulator.Run()
        end_of_simulation = max([communicator.get_current_time() for communicator in self.communicator_list])

        # for i, comm_matrix in enumerate(self.global_comm_matrix_list):
        #     print("---------%d-----------" % i)
        #     print(comm_matrix)

        if self.is_finishied():
            self._time_consuming = end_of_simulation - start_of_simulation
        else:
            self._time_consuming = -1
        return self._time_consuming
