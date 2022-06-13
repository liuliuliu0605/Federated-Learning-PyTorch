
from matplotlib import pyplot as plt
import numpy as np

y = [1,10,5,20,39,10,30,20,22,24,27,32,47,42,39,26,19]
x = range(len(y))

plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot([x[-1],x[-1]], [0, max(y)])

for degree in [2,3]:
    z1 = np.polyfit(x, y, degree)
    p1 = np.poly1d(z1)
    xx = []
    yvals = []
    value = 1
    i = 0
    while value > 0:
        xx.append(i)
        value = p1(i)
        yvals.append(value)
        print(value)
        i+=1
    plt.plot(xx, yvals, label='degree=%d'%degree)

# plt.xticks(xx)
plt.grid()
plt.legend()
plt.show()