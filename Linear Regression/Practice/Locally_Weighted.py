import numpy as np
import matplotlib.pyplot as plt

def local_weighted_regression(x0, X, Y, tau):
    x0_new = np.array([np.ones(len(x0)), x0.flatten()]).T

    xw = X.T * weights_calculate(x0, X, tau)
    best_theta_value = np.linalg.inv(xw.dot(X)).dot(xw.dot(Y))

    return best_theta_value.T.dot(x0_new)

def weights_calculate(x0_new, X, tau):
    return 

def main():
    n = 1000

    X = np.linspace(-3, 3, num=n)
    Y = np.ans(X ** 2 - 1)

    X += np.random.normal(scale=.1, size=n)

main()

