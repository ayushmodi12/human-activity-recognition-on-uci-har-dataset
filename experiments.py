import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
def generate_fake_data(N, M):
    data = np.random.choice([0, 1], size=(N, M))
    labels = np.random.choice([0, 1], size=N)
    return pd.DataFrame(data, columns=[f'{i}' for i in range(M)]), pd.Series(labels)

df=generate_fake_data(2,4)
print(df[1])

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
m=5
for n in range (50,501,50):
    tree = DecisionTree(criterion="information_gain",max_depth=3)
    X,y=generate_fake_data(n,m)
    t1=time.time()
    tree.fit(X,y)
    t2=time.time()
    print(t2-t1)
    t3=time.time()
    tree.predict(X)
    t4=time.time()
    print(t4-t3)
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
