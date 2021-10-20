from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

# print(X.shape, y.shape)

# knn = KNeighborsClassifier(n_neighbors = 5)

""" 
    Cv: cross-vaidation
    Steps for Cv
    Dataset is split into K fold of equal size
    Each fold acs as testing set 1 time, and acts as the training set k-1 times
"""
# scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
# print(scores.mean())

# k_range = range(1, 31)
# k_scores = []

# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors = k)
#     scores = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
#     k_scores.append(scores.mean())

# plt.plot(k_range, k_scores, color="blue")
# plt.xlabel("K values")
# plt.ylabel("Cross validation accuracy")

# plt.show()

knn = KNeighborsClassifier()

k_range = list(range(1, 31))
# print(k_range)

param_grid = dict(n_neighbors = k_range)
# print(param_grid)

grid = GridSearchCV(knn, param_grid, cv = 10, scoring='accuracy')
grid.fit(X, y)

# print(grid.best_score_)
# print(grid.best_params_)
# print(grid.best_estimator_)




