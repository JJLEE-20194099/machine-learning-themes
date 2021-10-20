import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as  sns

df = pd.read_csv('./data.csv')

df.dropna(axis = 1, how ='all', inplace = True)
y = df['diagnosis']
X = df.drop('diagnosis', axis = 1)
X = X.drop('id', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

K = []
training = []
test = []
scores = {}

for k in range(2, 21):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, Y_train)
    
    training_score = model.score(X_train, Y_train)
    test_score = model.score(X_test, Y_test)
    K.append(k)

    training.append(training_score)
    test.append(test_score)
    scores[k] = [training_score, test_score]


# ax = sns.stripplot(K, training)
# ax.set(xlabel = 'K', ylabel = 'Training Score')

# plt.show()

# ax = sns.stripplot(K, test)
# ax.set(xlabel = 'K', ylabel = 'Test Score')

# plt.show()

plt.scatter(K, training, color='k')
plt.scatter(K, test, color='g')
plt.show()