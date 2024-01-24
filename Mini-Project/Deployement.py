#Important Note: Make sure you have openxlsx library installed since it is used to read and extract data from .xlsx files from the directory. This can be done using !pip install openxlsx

import os
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from FeatureExtractor.extractor import extract #Self-made library to extract the necessary features we require

classes = {"Walking":1,"WalkingUpstairs":2,"WalkingDownstairs":3,"Sitting":4,"Standing":5,"Laying":6}  #Dictionary to define the classification of the activity

X_test_collected_ax = []
X_test_collected_ay = []
X_test_collected_az = []
y_test_collected = []
print("Training the recognizer...")

for folder in classes.keys():
    files = os.listdir("..\\Collected-Data\\"+folder)
    for file in files:
        path = "..\\Collected-Data\\" + folder + "\\" + file

        try: 
            df = pd.read_excel(path)
        except:
            df = pd.read_csv(path)
        
        df = df[0:500]
        df.columns = ['time','ax','ay','az','at']

        X_test_collected_ax.append(np.array(df['ax']))
        X_test_collected_ay.append(np.array(df['ay']))
        X_test_collected_az.append(np.array(df['az']))
        y_test_collected.append(classes[folder])

X_test_collected = []

#Making an 3D array to feed to extract to get the required features 
for i in range(len(X_test_collected_ay)):
    temp = []
    for j in range(500):
        temp.append([X_test_collected_ax[i][j],X_test_collected_ay[i][j],X_test_collected_az[i][j]])
    X_test_collected.append(temp)

X_test_collected = np.array(X_test_collected)

X_test_collected = extract(X_test_collected)  # Extract takes a 3D Array and returs an 2D array with the selected features
print("Shape of X_test_collected (Featurized):",X_test_collected.shape)


#Loading the data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')


#Appending everything into the X_Data
X_data = []
for i in range(len(X_test)):
    X_data.append(X_test[i])

for i in range(len(X_train)):
    X_data.append(X_train[i])

for i in range(len(X_val)):
    X_data.append(X_val[i])

X_train = extract(X_data)
y_train = np.array(list(y_test)+list(y_train)+list(y_val))


Recognizer = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=2, criterion='gini', random_state=43)
Recognizer.fit(X_train, y_train)

y_pred = Recognizer.predict(X_test_collected)
accuracy = accuracy_score(y_test_collected,y_pred)
print("Accuracy of the Decision Tree model is: ",accuracy*100,"%")
con_mat = confusion_matrix(y_test_collected,y_pred, labels=Recognizer.classes_)
print("Displaying Confusion Matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=con_mat,display_labels=Recognizer.classes_)
disp.plot()
plt.show()
            