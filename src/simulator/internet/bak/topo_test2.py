import numpy as np
node_num=9
matrix = (np.random.random((node_num, node_num)) > 0.5).astype("d")
for i in range(node_num):
    matrix[i, i] = 1

for i in range(node_num):
    for j in range(node_num):
        matrix[i, j] = matrix[j, i]

matrix=np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 1],
                 [0, 0, 1, 0, 0, 1, 0, 0, 0],
                 [1, 0, 0, 0, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0, 0, 0, 1, 0]])
print(matrix)
w_matrix = np.zeros_like(matrix, dtype=float)
for i in range(node_num):
    for j in range(node_num):
        if i != j and matrix[i, j] > 0:
            w_matrix[i, j] = 1. / (max(sum(matrix[:,i]), sum(matrix[:,j])+1))
            print(w_matrix[i, j])
    w_matrix[i, i] = 1 - w_matrix[i].sum()

# print(matrix)
print(w_matrix)