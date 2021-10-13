import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Linear_Regression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.b = [0., 1.]
    
    def update_coeffs(self, learning_rate):
        Y_pred = self.predcit()
        Y = self.Y
        m = len(Y)
        self.b[0] = self.b[0] - learning_rate * ((1/m) * np.sum(Y_pred - Y))
        self.b[1] = self.b[1] - learning_rate * ((1/m) * np.sum((Y_pred - Y) * self.X))

    def predcit(self, X=[]):
        if not X: X = self.X
        b = self.b
        Y_pred = np.array([b[0] + b[1] * x for x in X])
        return Y_pred

    def get_current_accuracy(self, Y_pred):
        p, e = Y_pred, self.Y
        n = len(Y_pred)
        
        numOfTrueCases = 0

        for i in range(n):
            if (abs(p[i] - e[i]) < 1e-5):
                numOfTrueCases += 1

        return numOfTrueCases / n



    def compute_cost(self, Y_pred):
        m = len(self.Y)
        J = (1/(2 * m)) * np.sum((Y_pred - self.Y) ** 2)
        return J

    def plot_best_fit(self, Y_pred, name):
        f = plt.figure(name)
        plt.scatter(self.X, self.Y, color='b')
        plt.plot(self.X, Y_pred, color='g')
        f.show()
    


def main():
        data = pd.read_csv('data.csv', header=0, sep=',').values
        N, d = data.shape
        X = data[:, 0]
        Y = data[:, 1]

        print(X, Y)


        regressor = Linear_Regression(X, Y)
        iterations = 0
        steps = 100
        learning_rate = 0.01
        costs = []



        while 1:
            Y_pred = regressor.predcit()
            cost = regressor.compute_cost(Y_pred)
            costs.append(cost)
            regressor.update_coeffs(learning_rate)

            iterations += 1

            if iterations % steps == 0:
                print(iterations, " epochs elapsed")
                print("Current accuracy is: ", regressor.get_current_accuracy(Y_pred))

                stop = input("Do you want to stop (y/n)??")
                if stop == "y":
                    break

        
        regressor.plot_best_fit(Y_pred, 'Final Best Fit Line')

        h = plt.figure('Verification')
        plt.plot(range(iterations), costs)
        h.show()    

main()







