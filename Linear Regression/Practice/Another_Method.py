import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

from sklearn.linear_model import LinearRegression

data = pd.read_csv('../Real Estate Price/data.csv', header=0, delimiter=',').values;
x = data[:, 0]
y = data[:, 1]

x_mean = x.mean()
y_mean = y.mean()

Sxx = sum((x - x_mean) * (x - x_mean))
Sxy = sum((x - x_mean) * (y - y_mean))
Syy = sum((y - y_mean) * (y - y_mean))

Slope = Sxy / Sxx;
Intercept = y_mean - Slope * x_mean

print('Slope: ', Slope)
print('Intercept: ', Intercept)

y_pred = Slope * x + Intercept;

plt.scatter(x, y, color='red')
plt.plot(x, y_pred, color='green')
plt.xlabel('Diện tích')
plt.ylabel('Giá')


# 100,1515.28

# x_test = 100
# y_test = 1515.28

# y_predict = x_test * Slope + Intercept;
# print(y_predict)

error = y - y_pred
square_error = sum(error * error)
print(square_error)

R2 = 1 - (square_error / Syy)
print(R2);

data = pd.read_csv('../Real Estate Price/data.csv', header=0, delimiter=',').values;

x = data[:, 0].reshape((-1, 1))
y = data[:, 1].reshape((-1, 1))

print(x.shape[0])

regression_model = LinearRegression()
regression_model.fit(x, y)

y_predict = regression_model.predict(x);
plt.plot(x, y_predict, color='yellow');

error = y - y_predict
se = sum(error ** 2);

n = x.shape[0]
mse = se / n
rmse = np.sqrt(mse)

y_mean = y.mean()
Syy = sum((y - y_mean) ** 2)[0]

r2 = 1 - se / Syy;
print(r2);
plt.show();

