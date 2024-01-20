import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

classes = {"Walking":1,"WalkingUpstairs":2,"WalkingDownstairs":3,"Sitting":4,"Standing":5,"Laying":6}

X_test_collected = []
y_test_collected = []
print("Training the recognizer...")
for folder in classes.keys():
    files = os.listdir("..\\Collected-Data\\"+folder)
    for file in files:
        path = "..\\Collected-Data\\" + folder + "\\" + file
        # print(path)
        try: 
            df = pd.read_excel(path)
        except:
            df = pd.read_csv(path)
        
        df = df[0:500]
        df.columns = ['time','ax','ay','az','at']
        X_test_collected.append(np.array(df['at'])**2)
        y_test_collected.append(classes[folder])

X_test_collected = np.array(X_test_collected)

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
# y_data = np.array(list(y_train)+list(y_val))
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

Recognizer = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=5, criterion='gini', random_state=42)
Recognizer.fit(X_train, y_train)

y_pred = Recognizer.predict(X_test_collected)
accuracy = accuracy_score(y_test_collected,y_pred)
print("Accuracy of the Decision Tree model is (max_depth == None): ",accuracy)
con_mat = confusion_matrix(y_test_collected,y_pred, labels=Recognizer.classes_)
print("Displaying Confusion Matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=con_mat,display_labels=Recognizer.classes_)
disp.plot()
plt.show()
            