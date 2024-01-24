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
# df = df.drop(['cylinders'], axis=1)
# df = df.drop(['origin'], axis=1)
df.columns = range(len(df.columns))
y = y.astype('float64')
tree = DecisionTree(criterion="information_gain",max_depth=5)  # Split based on Inf. Gain
tree.fit(df,y)

# print(tree.tree)
y_hat = tree.predict(df)
print("Results of our decision tree: ")
# tree.plot()
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))
# print(y_hat,y)



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(df,y)
Y_pred = regressor.predict(df)

# Calculate MSE and RMSE
mse = mean_squared_error(y, Y_pred)
rmse = sqrt(mse)
print("Results of sklearn decision tree: ")
# print(f"Mean Squared Error: {mse}")
print(f"RMSE: {rmse}")
print("MAE: ", mae(Y_pred, y))
# print(regressor.get_depth())