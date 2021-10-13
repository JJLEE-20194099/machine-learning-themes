import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

data = pd.read_csv('data.csv', header=0, sep=',').values

N, d = data.shape

x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, -1].reshape(-1, 1)

x_cho_vay = x[y[:, 0] == 1]
x_tu_choi = x[y[:, 0] == 0]

plt.scatter(x_cho_vay[:, 0], x_cho_vay[:, 1], c='red', edgecolor='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x_tu_choi[:, 0], x_tu_choi[:, 1], c='red', edgecolor='blue', edgecolors='none', s=30, label='tu choi')
plt.legend(loc=1)
plt.xlabel('muc luong (trieu)')
plt.ylabel('kinh nghiem(nam)')

x = np.hstack((np.ones((N, 1)), x))
w = np.array([0., 0.1, 0.1]).reshape(-1, 1)

numOfIterations = 100000
cost = np.zeros((numOfIterations, 1))
learning_rate = 0.01

time = []

for i in range(0, numOfIterations):
    time.append(i)
    y_predcit = sigmoid(np.dot(x, w))
    cost[i] = -np.sum(np.multiply(y, np.log(y_predcit)) + np.multiply(1 - y, np.log(1 - y_predcit)))

    w = w - learning_rate * np.dot(x.T, y_predcit - y)

t = 0.5
plt.plot((4, 10), (-(w[0] + w[1] * 4 + np.log(1/t - 1)) / w[2], -(w[0] + w[1] * 10 + np.log(1/t - 1)) / w[2]), 'g')
# plt.plot(time, cost)
plt.show()

np.save('weight logistic.npy', w)
w = np.load('weight logistic.npy')
print(w)

