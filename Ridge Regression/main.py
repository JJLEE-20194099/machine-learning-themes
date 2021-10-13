# import libraries

import numpy as np
import pandas as pd
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RidgeRegression():
    def __init__( self, learning_rate, iterations, l2_penalty):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty
    
    
    # Function for model training

    def fit( self, X, Y ):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape

        # weight initailization

        self.W = np.zeros( self.n )

        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        
        for i in range(self.iterations):
            self.update_weights()
        
        return self
    
    # Helper function to update weights in gradient descent

    def update_weights( self ):
        Y_pred = self.predict( self.X )

        # calculate gradients

        dW = ( -(2 * (self.X.T).dot(self.Y - Y_pred))
            + 2 * self.l2_penalty * self.W ) / self.m

        db = -2 * np.sum(self.Y - Y_pred) / self.m

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        
        return self
    
    def predict( self, X):
        return X.dot(self.W) + self.b
    

def main():

    # import dataset
    df = pd.read_csv('./salary_data.csv')
    
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

    Y_mean = np.mean(Y_test)
    
    y_variance = np.sum((Y_test - Y_mean) ** 2 )

    model = RidgeRegression( iterations = 1000, learning_rate = 0.01, l2_penalty = 1)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    se = np.sum((Y_test - Y_pred) ** 2)

    r2 = 1 - se / y_variance

    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.title('Salary vs Experience')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary vs Experience')
    plt.show()

    print(r2)
    
if __name__ == '__main__':
        main()

    
    


        
