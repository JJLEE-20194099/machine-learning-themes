import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv', header=0, sep=',').values
N = data.shape[0]

x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel("Diện tích")
plt.ylabel("Giá")

x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 1.]).reshape(-1, 1)

numOFfIterations = 10000
cost = np.zeros((numOFfIterations, 1))
learning_rate = 0.00001

for i in range(1, numOFfIterations):
    r = np.dot(x, w) - y
    cost[i] = 0.5 * np.sum(np.multiply(r, r))
    w[0] -= learning_rate * np.sum(r)
    w[1] -= learning_rate * np.sum(np.multiply(r, x[:,1].reshape(-1, 1)))
    # print(cost[i])  

predict = np.dot(x, w)
# plt.plot((x[0][1], x[N - 1]), (predict[0], predict[N - 1]), 'r')
plt.show()
x1=63
y1= w[0] + w[1] * x1

print(y1)

