import math

# cluster groups
# users_groups = [[1, 2, 9, 13, 24, 26, 37, 47, 55, 60, 61, 64, 71, 72, 82, 95, 98],
#                 [0, 5, 6, 11, 14, 19, 21, 36, 42, 43, 45, 53, 57, 73, 75, 77, 85, 92, 94],
#                 [10, 12, 16, 18, 20, 29, 33, 34, 39, 40, 44, 48, 52, 54, 56, 58, 62, 63, 65, 69, 79, 81, 84, 87, 96, 97, 99],
#                 [17, 23, 27, 28, 31, 35, 38, 41, 51, 59, 66, 67, 68, 70, 74, 76, 86, 88, 90, 91],
#                 [3, 4, 7, 8, 15, 22, 25, 30, 32, 46, 49, 50, 78, 80, 83, 89, 93]]
users_groups = [[1, 4, 9, 10, 13, 17, 19, 20, 52, 54, 60, 61, 64, 66, 68, 71, 76, 82, 89, 98],
                [6, 7, 31, 33, 34, 35, 36, 39, 40, 43, 44, 55, 59, 62, 69, 70, 78, 92, 93, 99],
                [11, 14, 21, 25, 26, 30, 46, 49, 53, 58, 65, 72, 74, 79, 81, 85, 87, 90, 91, 96],
                [0, 8, 16, 22, 27, 29, 32, 41, 45, 47, 48, 57, 67, 73, 77, 83, 84, 86, 88, 97],
                [2, 3, 5, 12, 15, 18, 23, 24, 28, 37, 38, 42, 50, 51, 56, 63, 75, 80, 94, 95]]


# cluster locations
centroids = [[50, 225], [25, 110], [125, 20], [220, 80], [200, 225]]
# centroids = [(35.18427536993528, 38.80790026650002),
#              (60.17708251481465, 208.11418793203705),
#              (195.41325010231304, 190.80862362840043),
#              (174.85768645158228, 59.7727107950872),
#              (37.64266505202888, 133.2751288730596)]

########################################### WSN ###########################################

energy_budget = 500  # energy budget, Joule
NB_CLUSTERS = 5
NB_NODES = 100
MSG_LENGTH = 7840 * 32  # bits, cnn: 1,984,192  , lr: 7840 * 32
LOCAL_UPDATES = 10  # local updates
routing_topology = 'FCM'
INITIAL_ENERGY = 50  # initial energy of each node, Joule
AREA_WIDTH = 250.0
AREA_LENGTH = 250.0
BS_POS_X = 125.0
BS_POS_Y = 125.0
BSID = -1
FUZZY_M = 1.5  # FCM fuzzyness coeficient
TRACE_ENERGY = 1
TRACE_ALIVE_NODES = 1
TRACE_COVERAGE = 1
TRACE_LEARNING_CURVE = 0

# Energy Configurations
# energy dissipated at the local updates (/bit)
E_UPDATE = 5e-10  # Joules
# energy dissipated at the transceiver electronic (/bit)
E_ELEC = 50e-9  # Joules
# energy dissipated at the power amplifier (supposing a multi-path fading channel) (/bin/m^4)
E_MP = 0.0013e-12  # Joules
# energy dissipated at the power amplifier (supposing a line-of-sight free-space channel (/bin/m^2)
E_FS = 10e-12  # Joules
# energy dissipated at the data aggregation (/bit)
E_DA = 5e-9  # Joules
THRESHOLD_DIST = math.sqrt(E_FS / E_MP)  # meters

INFINITY = float('inf')  # 正无穷
MINUS_INFINITY = float('-inf')  # 负无穷


# internet

