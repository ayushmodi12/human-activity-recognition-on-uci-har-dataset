'''
This code was inspired by Nipun Batra sir's class notebooks.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tsfel import time_series_features_extractor, get_features_by_domain
from sklearn.model_selection import train_test_split
from sklearn import tree
from FeatureExtractor.extractor import extract
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# X_data = []
# for i in range(len(X_test)):
#     temp = []
#     for j in range(len(X_test[0])):
#         temp.append(np.dot(X_test[i][j],np.transpose(X_test[i][j])))
#     X_data.append(temp)

# for i in range(len(X_train)):
#     temp = []
#     for j in range(len(X_train[0])):
#         temp.append(np.dot(X_train[i][j],np.transpose(X_train[i][j])))
#     X_data.append(temp)

# for i in range(len(X_val)):
#     temp = []
#     for j in range(len(X_val[0])):
#         temp.append(np.dot(X_val[i][j],np.transpose(X_val[i][j])))
#     X_data.append(temp)

# X_data = np.array(X_data)
X_data = extract(np.concatenate((X_test,X_train,X_val)))
print("X_data shape:",X_data.shape)
y_data = np.array(list(y_test)+list(y_train)+list(y_val))

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
print("Number of training examples: {}".format(len(X_train)))
print("Number of validation examples: {}".format(len(X_val)))

hyperparameters = {}
hyperparameters['max_depth'] = [1,2,3,4,5,6,7,8,9,10]
hyperparameters['min_samples_split'] = [2,3,4,5,6,7,8]
hyperparameters['criteria_values'] = ['gini', 'entropy']

best_accuracy = 0
best_hyperparameters = {}

out = {}
count = 0
for max_depth in hyperparameters['max_depth']:
    for min_samples_split in hyperparameters['min_samples_split']:
        for criterion in hyperparameters['criteria_values']:
            # Create and fit the decision tree classifier with the current hyperparameters
            Recognizer = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion, random_state=42)
            Recognizer.fit(X_train, y_train)
            
            # Evaluate the performance on the validation set
            val_accuracy = Recognizer.score(X_val, y_val)
            out[count] = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion, 'val_accuracy': val_accuracy}
            count += 1

hparam_df = pd.DataFrame(out).T
best_hyperparameters_row_id = 0
for i in range(0,len(hyperparameters['max_depth'])*len(hyperparameters['min_samples_split'])*len(hyperparameters['criteria_values'])):
    if (hparam_df['val_accuracy'][i] > hparam_df['val_accuracy'][best_hyperparameters_row_id]):
        best_hyperparameters_row_id = i

best_accuracy = hparam_df['val_accuracy'][best_hyperparameters_row_id]
best_hyperparameters = {'max_depth':hparam_df['max_depth'][best_hyperparameters_row_id], 'min_samples_split':hparam_df['min_samples_split'][best_hyperparameters_row_id], 'criterion':hparam_df['criterion'][best_hyperparameters_row_id] }
print("Best Hyperparameters:", best_hyperparameters)
print("Validation Set accuracy: {:.4f}".format(best_accuracy))