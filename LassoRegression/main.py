# import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LassoRegression():
    def __init__(self, learning_rate, iterations, l1_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l1_penalty = l1_penalty
    
    def fit(self, X, Y):
        self.m, self.n = X.shape

        self.W = np.zeros(self.m)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()

        return self
    
    def update_weights( self ):

        Y_pred = self.predict(self.X)

        dW = np.zeros( self.n)

        for j in range(self.n):
            if self.W[j] > 0:
                dW[j] = (-(2 * self.X[:, j].T).dot(self.Y - Y_pred) + self.l1_penalty) / self.m
            
            else :
                dW[j] = (-(2 * self.X[:, j].T).dot(self.Y - Y_pred) - self.l1_penalty) / self.m

        db = -2 * np.sum((self.Y - Y_pred)) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        return X.dot(self.W) + self.b

def main():
    df = pd.read_csv('./salary_data.csv')

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

    model = LassoRegression(iterations=1000, learning_rate=0.01, l1_penalty=500)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    plt.scatter(X_test, Y_test, color='blue')
    plt.scatter(X_test, Y_pred, color='orange')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

if (__name__ == '__main__'):
    main()


