"""
@Description : 
@File        : hypercube.py
@Project     : Federated-Learning-PyTorch
@Time        : 2022/4/11 22:45
@Author      : Xuezheng Liu
"""

n = 2
p = 2**n

for i in range(1, p):
    print([(node, node ^ i) for node in range(p)])