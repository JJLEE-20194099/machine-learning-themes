import numpy as np
import matplotlib.pyplot as plt

x = 1.0
numOfIterations = 10000
learning_rate = 0.0001

cost = np.zeros((numOfIterations,))

def calculate(x):
    return x * x  + 2 * x + 5

X = []
Y = []

for i in range(numOfIterations):
    curr_value = calculate(x)
    X.append(x)
    Y.append(curr_value)
    x -= (2 * x + 2) * learning_rate
    cost[i] = np.absolute(calculate(x) - curr_value)
    if (cost[i] < 1e-100):
        break

plt.plot(X, Y)

print (calculate(x))
    
plt.show()
