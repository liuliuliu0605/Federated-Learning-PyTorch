import ns.core
import numpy as np

from src.simulator.internet.communicator import Communicator
from collections import OrderedDict

CHECK_TIME_SLOT = 0.0001
DEFAULT_TIME_SLOT = 0.1
DEFAULT_OFFLINE_NUMBER = 0
MAX_BLOCK_DURATION = 9999999999999
BASE_PORT = 5000

class DecentralizedConsensus:

    def __init__(self, model_size, ps_node_container, matrix, data_rate=1e9, packet_size=536,
                 protocol='tcp', port=9, verbose=False, offline_params={}, max_block_duration=None):
        assert protocol in ['tcp', 'udp']
        assert ps_node_container.GetN() == len(matrix)

        self.ps_node_container = ps_node_container
        self.model_size = model_size
        self.matrix = matrix  # np.ndarray
        self.data_rate = data_rate
        self.packet_size = packet_size

        self.protocol = protocol
        self.port = port

        self.node_num = self.ps_node_container.GetN()
        self.verbose = verbose
        self.offline_params = offline_params
        # self.sender_receiver_matrix = None

        self._time_consuming = 0
        self.communicator_list = None
        self.global_comm_matrix_list = None
        self.max_block_duration = max_block_duration if max_block_duration is not None else MAX_BLOCK_DURATION

    def __del__(self):
        self.reset()

    def reset(self):
        self._time_consuming = 0
        for communicator in self.communicator_list:
            communicator.reset()

    def get_time_consuming(self):
        return self._time_consuming

    def init_app(self, start_time=0, stop_time=None, phases=1):
        # source node src (ps_node_container.Get(src)) will send data to
        # sink node dst (ps_node_container.Get(dst)) with "src+BASE_PORT" port listened

        self.global_comm_matrix_list = [np.zeros((self.node_num, self.node_num), dtype=int) for _ in range(phases)]
        self.communicator_list = [Communicator(self.ps_node_container.Get(i), i, self.offline_params,
                                               self.protocol, self.verbose, self.global_comm_matrix_list)
                                  for i in range(self.node_num)]
        self.sender_receiver_matrix = [[None for _ in range(self.node_num)] for _ in range(self.node_num)]

        for src in range(self.node_num):
            for dst in range(self.node_num):
                if src != dst and self.matrix[src, dst] > 0:
                    if self.verbose:
                        print("Mixing model: PS %d -> PS %d" % (src, dst))
                    app_receiver = self.communicator_list[dst].add_app_receiver(src, self.model_size, phases, src+BASE_PORT,
                                                                                start_time, stop_time)
                    dst_node = self.ps_node_container.Get(dst)
                    app_sender = self.communicator_list[src].add_app_sender(dst, dst_node, self.model_size, phases, src+BASE_PORT,
                                                                            self.packet_size, self.data_rate,
                                                                            start_time, stop_time)
                    self.communicator_list[dst].associate_upstream_app_sender(src, app_sender)
                    # self.sender_receiver_matrix[src][dst] = (app_sender, app_receiver)

    def is_finished(self):
        communicator_states = [communicator.is_finished() for communicator in self.communicator_list]
        return np.all(communicator_states)

    def offline_thread(self):
        # recover the nodes offline in the last time slot
        online_comms = [self.communicator_list[i] for i in range(len(self.communicator_list)) if
                        self.communicator_list[i].is_offline()]
        online_comm_ids = [comm.get_id() for comm in online_comms]

        # decide the nodes offline in this time slot
        offline_comms = np.random.choice(self.communicator_list,
                                         int(self.offline_params.get("number", DEFAULT_OFFLINE_NUMBER)), replace=False)
        offline_comm_ids = [comm.get_id() for comm in offline_comms]
        print(offline_comm_ids)
        # take out the nodes which will keep offline in this slot
        online_comms = [self.communicator_list[i] for i in online_comm_ids if i not in offline_comm_ids]
        online_comm_ids = [comm.get_id() for comm in online_comms]

        if self.verbose:
            print("\n---------------------------------------------------------------------------")
            print("[offline] At time %.6f %d nodes are:" % (ns.core.Simulator.Now().GetSeconds(), len(offline_comms)),
                  offline_comm_ids)
            print("[online] At time %.6f %d nodes are:" % (ns.core.Simulator.Now().GetSeconds(), len(online_comms)),
                  online_comm_ids)

        for i in range(len(offline_comms)):
            offline_comms[i].offline_operation()

        for i in range(len(online_comms)):
            online_comms[i].online_operation()
            online_comms[i].send_message()
            online_comms[i].inform_upstream_send_message()

        if not self.is_finished():
            ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(self.offline_params.get("time_slot", DEFAULT_TIME_SLOT))),
                                       self.offline_thread)

    def is_blocked_by(self, comm_x, comm_y):
        if self.matrix[comm_y.get_id(), comm_x.get_id()] <= 0:
            return False
        lagging_list = comm_x.get_lagging_communicator_ids()
        if len(lagging_list) == 1 and lagging_list[0] == comm_y.get_id():
            return True
        else:
            return False

    def unblock_thread(self):
        for src, communicator_y in enumerate(self.communicator_list):
            if communicator_y.is_offline():
                dst_list = [dst for dst in self.matrix[communicator_y.get_id(), :].nonzero()[0] if dst != communicator_y.get_id()]
                for dst in dst_list:
                    communicator_x = self.communicator_list[dst]
                    if self.is_blocked_by(communicator_x, communicator_y):
                        if self.verbose:
                            print("Node %d was blocked by node %d" % (communicator_x.get_id(), communicator_y.get_id()))
                        # unblock the online communicator y if exceeding maximum block duration
                        if ns.core.Simulator.Now().GetSeconds() - communicator_x.get_current_time() > self.max_block_duration:
                            ignoring_phase = communicator_x.abandon_data_from(communicator_y)
                            if ignoring_phase is not None:
                                _ = communicator_y.abandon_data_from(communicator_x, ignoring_phase)
                            # if self.verbose:
                            #     print("Node %d would not receive data from node %d in %d-th phase"
                            #           % (communicator_x.get_id(), communicator_y.get_id(), ignoring_phase))
                if len(dst_list) == 0:
                    src_list = self.matrix[:, communicator_y.get_id()].nonzero()[0]
                    for src in src_list:
                        communicator_x = self.communicator_list[src]
                        if ns.core.Simulator.Now().GetSeconds() - communicator_x.get_current_time() > self.max_block_duration:
                            ignoring_phase = communicator_y.abandon_data_from(communicator_x)

        if not self.is_finished():
            ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(CHECK_TIME_SLOT)), self.unblock_thread)

    def run(self, start_time, stop_time, phases=1):
        self.init_app(start_time, stop_time, phases)

        # any communicator may be offline in the initial phase
        ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(start_time)), self.offline_thread)
        for i in range(len(self.communicator_list)):
            ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(start_time)), self.communicator_list[i].send_message)

        # dynamically check whether any node are blocked by offline nodes
        ns.core.Simulator.Schedule(ns.core.Time(ns.core.Seconds(start_time)), self.unblock_thread)

        start_of_simulation = ns.core.Simulator.Now().GetSeconds() + start_time
        ns.core.Simulator.Run()
        # TODO: get last received time ?
        end_of_simulation = max([communicator.get_current_time() for communicator in self.communicator_list])

        if self.is_finished():
            self._time_consuming = end_of_simulation - start_of_simulation
        else:
            self._time_consuming = -1
        # print(self.get_comm_matrix_list())
        return self._time_consuming

    def get_comm_matrix_list(self):
        rs = []
        c=0
        for comm_matrix in self.global_comm_matrix_list:

            matrix = comm_matrix.copy()
            for i in range(self.node_num):
                for j in range(i, self.node_num):
                    if i == j:
                        matrix[i, j] = 1
                    elif matrix[i, j] != matrix[j, i]:
                        matrix[i, j] = matrix[j, i] = 0

            w_matrix = np.zeros_like(matrix, dtype=float)
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if i != j and matrix[i, j] > 0:
                        w_matrix[i, j] = 1 / (max(sum(matrix[:, i]), sum(matrix[:, j])))
                w_matrix[i, i] = 1 - w_matrix[i].sum()

            W = w_matrix - np.ones_like(w_matrix) / self.node_num
            eigen, _ = np.linalg.eig(np.matmul(W, W.T))
            p = 1 - np.max(eigen)
            tmp = {}
            tmp['comm'] = comm_matrix.tolist()
            tmp['weight'] = w_matrix.tolist()
            tmp['p'] = float(p)
            rs.append(tmp)
            c+=1
        return rs


    # def laplace(self, matrix):
    #     node_num = len(matrix)
    #     for i in range(node_num):
    #         for j in range(node_num):
    #             if i == j:
    #                 matrix[i, j] = 1
    #
    #     w_matrix = np.zeros_like(matrix, dtype=np.float32)
    #     matrix_sum = matrix.sum(1)
    #     for i in range(node_num):
    #         w_matrix[i, i] = matrix_sum[i]
    #     laplace_matrix = w_matrix - matrix
    #
    #     max_alpha = 1 / laplace_matrix.max()
    #     w_matrix = np.identity(node_num) - laplace_matrix * max_alpha
    #     max_p = 1 - np.linalg.norm(w_matrix - 1 / node_num, ord=2) ** 2
    #     for alpha in np.arange(0, max_alpha, 0.01):
    #         tmp = np.identity(node_num) - laplace_matrix * alpha
    #         p = 1 - np.linalg.norm(tmp - 1 / node_num, ord=2) ** 2
    #         if p > max_p:
    #             max_p = p
    #             w_matrix = tmp
    #     return w_matrix




