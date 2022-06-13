import config as cf
import logging


class EnergySource(object):
    def __init__(self, parent):
        self.energy = cf.INITIAL_ENERGY
        self.node = parent

    def recharge(self):
        self.energy = cf.INITIAL_ENERGY


class Battery(EnergySource):
    def consume(self, energy, type='default'):

        if self.energy >= energy:
            self.energy -= energy
            self.node.network_handler.energy_dis[type] += energy
        else:
            logging.info("node %d: battery is depleted." % (self.node.id))
            self.energy = 0
            self.node.battery_depletion()


class PluggedIn(EnergySource):
    def consume(self, energy, type='default'):
        pass
