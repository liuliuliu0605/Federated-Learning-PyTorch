import numpy as np

M = 1000000000
d = 10
r = 1-1/M**(2/d)
print(1/np.log2(1/r))