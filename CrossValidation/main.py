from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

