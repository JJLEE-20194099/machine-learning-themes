import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

x, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=10, random_state=10)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.scatter(x, y, marker='o')

y = y.reshape((100, 1))

x_new = np.array([np.ones(len(x)), x.flatten()]).T

theta_best_value = np.linalg.inv(x_new.T.dot(x_new)).dot(x_new.T.dot(y))

x_sample = np.array([[-2], [4]])
x_sample_new = np.array([np.ones(len(x_sample)), x_sample.flatten()]).T
predict_sample = x_sample_new.dot(theta_best_value)

lr = LinearRegression()
lr.fit(x, y)
standard_predict_sample = lr.predict(x_sample)


plt.plot(x_sample, predict_sample, c='red')
# plt.plot(x_sample, standard_predict_sample, c='green')
plt.show()

