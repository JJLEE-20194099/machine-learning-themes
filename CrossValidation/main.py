from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
iris = load_iris()

X = iris.data
y = iris.target

# train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 6)

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(metrics.accuracy_score(y_test, y_pred))


"""
    Comparing cross-vaidation to train/test 

    Advantages of cross-validation:
        + More accurate estimate of out-of-sample accuracy
        + More efficient use of data
            -> every observation is used for both training and testing
        
    Advantages of train/test split:

        + Run k times faster than k-fold cross-validation
            -> this is because K_fold cross-vaidation repeats the train/test split k-times
        
        + Simple to examine the detailed resuslts of the testing process

"""

k_ranges = list(range(1, 31))
# k_scores = []

# for i in k_ranges:
#     knn = KNeighborsClassifier(n_neighbors = i)
#     scores = cross_val_score(knn, X, y, cv = 10, scoring="accuracy")
#     k_scores.append(scores.mean())

# plt.plot(k_ranges, k_scores)
# plt.show()

knn = KNeighborsClassifier()

k_range = list(range(1, 31))

param_dict = dict(n_neighbors = k_range)
grid = GridSearchCV(knn, param_dict, cv = 10, scoring = 'accuracy')
grid.fit(X, y)

knn = KNeighborsClassifier(n_neighbors = k_range[grid.best_index_])

scores = cross_val_score(knn, X, y, cv=10, scoring = 'accuracy')
print(scores.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 6)

knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))





