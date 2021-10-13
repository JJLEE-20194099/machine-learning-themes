import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data.csv', header=0, sep=',').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

plt.scatter(x, y)
plt.xlabel('Diện tích')
plt.ylabel('Giá')

Lrg = LinearRegression()
Lrg.fit(x, y)
y_pred = Lrg.predict(x)
plt.plot((x[0], x[-1]), (y_pred[0], y_pred[-1]), 'r')

plt.show()

