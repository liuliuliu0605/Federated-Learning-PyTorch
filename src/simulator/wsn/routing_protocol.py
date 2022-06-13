"""This class defines the interface that should be used when defining a new
routing protocol.
"""

class RoutingProtocol(object):

    def initialize(self, network):
        pass

    def setup_phase(self, network, round_nb):
        """This method is called before every round. It only redirects to
        protected methods."""
        pass
