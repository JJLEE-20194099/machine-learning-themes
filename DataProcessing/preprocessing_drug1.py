import numpy as np
import pandas as pd
from sklearn import preprocessing

my_data = pd.read_csv("./data/drug200.csv", delimiter=",")
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values;

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder();
le_BP.fit(['LOW', 'HIGH', 'NORMAL'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder();
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

mean = X[:, -1].mean()
std = X[:, -1].std()
X[:, -1] = (X[:, -1] - mean) / std

print(X[:5])

y = my_data["Drug"]
y = y.values;

le_DrugTarget = preprocessing.LabelEncoder();
le_DrugTarget.fit(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
y = le_DrugTarget.transform(y);
y = y.reshape((-1, 1))








