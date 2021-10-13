import numpy as np;
import pandas as pd;

my_data = pd.read_csv("../data/drug200.csv", delimiter=",")
print(my_data["Sex"].unique())
print(my_data["Sex"].value_counts())


one_hot_coded_data = pd.get_dummies(my_data, columns=["Sex", "BP", "Cholesterol"])
print(one_hot_coded_data)


