import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

df = pd.DataFrame(data)
df = df.replace(['?'], np.nan)
df = df.dropna()
df = df.drop(['car name'], axis=1)
df = df.astype('float64')
y = df['mpg']
df = df.drop(['mpg'], axis=1)
df = df.drop(['cylinders'], axis=1)
# df = df.drop(['origin'], axis=1)
df.columns = range(len(df.columns))
y = y.astype('float64')
tree = DecisionTree(criterion="information_gain",max_depth=6)  # Split based on Inf. Gain
tree.fit(df,y)

print(tree.tree)
y_hat = tree.predict(df)
# tree.plot()
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))
print(y_hat,y)