import ns.core
import ns.network

class Flow:

    def __init__(self, total_tx_bytes, write_size=1040, protocol='tcp'):
        assert protocol in ['tcp', 'udp']

        self.protocol = protocol
        self.total_tx_bytes = total_tx_bytes
        self.current_tx_bytes = 0
        self.write_size = write_size

    def start_flow(self, src_socket, dst_ip, dst_port, verbose=False):
        if self.protocol == 'tcp':
            ns.core.Simulator.ScheduleNow(self._tcp_flow, src_socket, dst_ip, dst_port, verbose)
        else:
            raise NotImplementedError()

    def _tcp_flow(self, src_socket, dst_ip, dst_port, verbose=False):

        if verbose:
            ns.core.NS_LOG_LOGIC("Starting flow at time ", ns.core.Simulator.Now().GetSeconds())

        src_socket.Connect(ns.network.InetSocketAddress(dst_ip, dst_port))

        def write_until_buffer_full(src_socket, tx_space):
            while self.current_tx_bytes < self.total_tx_bytes and src_socket.GetTxAvailable() > 0:
                left = self.total_tx_bytes - self.current_tx_bytes
                data_offset = self.current_tx_bytes % self.write_size
                to_write = self.write_size - data_offset
                to_write = min(to_write, left)
                to_write = min(to_write, src_socket.GetTxAvailable())
                packet = ns.network.Packet(to_write)
                amount_sent = src_socket.Send(packet, 0)
                if amount_sent < 0:
                    return
                self.current_tx_bytes += amount_sent

            if self.current_tx_bytes >= self.total_tx_bytes:
                src_socket.Close()

        src_socket.SetSendCallback(write_until_buffer_full)
        write_until_buffer_full(src_socket, src_socket.GetTxAvailable())


