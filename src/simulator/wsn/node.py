from .utils import *
import numpy as np
import config as cf
from .energy_source import Battery, PluggedIn
import logging


class Node(object):
    def __init__(self, id, parent=None, loc=None):

        if loc is None:
            self.pos_x = np.random.uniform(0, cf.AREA_WIDTH)
            self.pos_y = np.random.uniform(0, cf.AREA_LENGTH)
        else:
            self.pos_x, self.pos_y = loc

        if id == cf.BSID:
            self.energy_source = PluggedIn(self)
        else:
            self.energy_source = Battery(self)

        self.id = id
        self.network_handler = parent

        self.reactivate()

    def reactivate(self):
        """Reactivate nodes for next simulation."""
        self.alive = 1
        self.tx_queue_size = 0
        self._next_hop = cf.BSID
        self.distance_to_endpoint = 0
        self.amount_sensed = 0
        self.amount_transmitted = 0
        self.amount_received = 0
        self.membership = cf.BSID
        # aggregation function determines the cost of forwarding messages
        # (in number of bits)
        self.aggregation_function = lambda x: 0
        self.time_of_death = cf.INFINITY
        self._is_sleeping = 0
        self.sleep_prob = 0.0
        # for coverage purposes
        self.neighbors = []
        self.nb_neighbors = -1
        self.exclusive_radius = 0

    @property
    def next_hop(self):
        return self._next_hop

    @next_hop.setter
    def next_hop(self, value):
        self._next_hop = value
        distance = calculate_distance(self, self.network_handler[value])
        self.distance_to_endpoint = distance

    def _only_active_nodes(func):
        """This is a decorator. It wraps all energy consuming methods to
        ensure that only active nodes execute this method. Also it automa-
        tically calls the battery.
        """

        def wrapper(self, *args, **kwargs):
            if self.alive:
                func(self, *args, **kwargs)
                return 1
            else:
                return 0

        return wrapper

    def is_head(self):
        if self.next_hop == cf.BSID and self.id != cf.BSID and self.alive:
            return 1
        return 0

    def is_ordinary(self):
        return 1 if self.next_hop != cf.BSID and self.id != cf.BSID else 0

    @_only_active_nodes
    def transmit(self, msg_length, destination=None, type='default'):
        logging.debug("node %d transmitting." % (self.id))

        if not destination:
            destination = self.network_handler[self.next_hop]
            distance = self.distance_to_endpoint
        else:
            distance = calculate_distance(self, destination)

        # transmitter energy model
        energy = cf.E_ELEC
        if distance > cf.THRESHOLD_DIST:
            energy += cf.E_MP * (distance ** 4)
        else:
            energy += cf.E_FS * (distance ** 2)
        energy *= msg_length

        # automatically call other endpoint receive
        destination.receive(msg_length, type)
        self.amount_transmitted += msg_length

        self.energy_source.consume(energy, type)

    @_only_active_nodes
    def broadcast(self, msg_length, destination_list=None, type='default'):
        distance_list = []
        if destination_list is None:
            destination_list = self.network_handler.get_alive_nodes()
        destination_list = [dest for dest in destination_list if dest.id != self.id]
        if len(destination_list) == 0:
            return

        logging.debug("node %d broadcasting." % (self.id))

        for destination in destination_list:
            distance = calculate_distance(self, destination)
            distance_list.append(distance)

        max_distance = max(distance_list)

        # transmitter energy model
        energy = cf.E_ELEC
        if max_distance > cf.THRESHOLD_DIST:
            energy += cf.E_MP * (max_distance ** 4)
        else:
            energy += cf.E_FS * (max_distance ** 2)
        energy *= msg_length

        # automatically call other endpoint receive
        for destination in destination_list:
            destination.receive(msg_length, type)
        self.amount_transmitted += msg_length

        self.energy_source.consume(energy, type)

    @_only_active_nodes
    def receive(self, msg_length, type='default'):
        logging.debug("node %d receiving." % (self.id))
        self.amount_received += msg_length

        # energy model for receiver
        energy = cf.E_ELEC * msg_length
        self.energy_source.consume(energy, type)

    @_only_active_nodes
    def update(self, msg_length, k=1):
        logging.debug("node %d local update." % (self.id))

        for _ in range(k):
            # energy for one local update
            energy = cf.E_UPDATE * msg_length
            self.energy_source.consume(energy, 'local-update')

    def battery_depletion(self):
        self.alive = 0
        self.sleep_prob = 0.0
        self.time_of_death = self.network_handler.round
        self.network_handler.deaths_this_round += 1
