from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

logreg = LogisticRegression()
logreg.fit(x, y)

w = np.zeros((d, 1))

w[0, 0] = logreg.intercept_
w[1:, 0] = logreg.coef_

t=0.5
plt.plot((4, 10), (-(w[0] + w[1] * 4 + np.log(1/t - 1)) / w[2], -(w[0] + w[1] * 10 + np.log(1/t - 1)) / w[2]), 'g')
plt.show()


