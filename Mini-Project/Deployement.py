import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}
collected_data_dir = r"Collected-Data"
folder = "WalkingDownstairs"
file = "Subject_1.csv"
X_test = []

df = pd.read_excel(r"D:\mithil\IIT GN\Year_3\Semester 2\Machine Learning\assignment-1-ml-doofenshmirtz-evil-inc\Collected-Data\WalkingDownstairs\Subject_1.xlsx")
X_test_collected = np.array(df['at'])


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

X_data = []
for i in range(len(X_test)):
    temp = []
    for j in range(len(X_test[0])):
        temp.append(np.dot(X_test[i][j],np.transpose(X_test[i][j])))
    X_data.append(temp)

for i in range(len(X_train)):
    temp = []
    for j in range(len(X_train[0])):
        temp.append(np.dot(X_train[i][j],np.transpose(X_train[i][j])))
    X_data.append(temp)

for i in range(len(X_val)):
    temp = []
    for j in range(len(X_val[0])):
        temp.append(np.dot(X_val[i][j],np.transpose(X_val[i][j])))
    X_data.append(temp)

X_data = np.array(X_data)
y_data = np.array(list(y_test)+list(y_train)+list(y_val))

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

Recognizer = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=5, criterion='gini', random_state=42)
Recognizer.fit(X_train, y_train)
X_test_collected = X_test_collected.reshape((1,500))
y_pred = Recognizer.predict(X_test_collected)
print(y_pred)
            