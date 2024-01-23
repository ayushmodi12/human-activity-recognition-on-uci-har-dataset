import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.7)
X_train=pd.DataFrame(X_train)
y_train=pd.Series(y_train)
X_test=pd.DataFrame(X_test)
y_test=pd.Series(y_test)


#Q2 (A)
for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X_train, y_train)
    # print(tree.tree)
    y_hat = tree.predict(X_test)
    # tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    for cls in y_test.unique():
        print(f"Class: {cls}")
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))

#Q2(B)
        
def k_fold(X,y,criteria,depth,k=5)->int:
    # Initialize lists to store predictions and accuracies

    predictions = {}
    accuracies = []

    # Calculate the size of each fold
    fold_size = len(X) // k

    # Perform k-fold cross-validation
    for i in range(k):
        # Split the data into training and test sets
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]
        
        training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)
        training_set=pd.DataFrame(training_set)
        training_labels=pd.Series(training_labels)
        # Train the model
        dt_classifier =DecisionTree(criterion=criteria,max_depth=depth)
        dt_classifier.fit(training_set, training_labels)
        test_set=pd.DataFrame(test_set)
        # Make predictions on the validation set
        fold_predictions = dt_classifier.predict(test_set)
        test_labels=pd.Series(test_labels)
        # Calculate the accuracy of the fold
        fold_accuracy =accuracy(fold_predictions,test_labels)
        # Store the predictions and accuracy of the fold
        predictions[i] = fold_predictions
        accuracies.append(fold_accuracy)

    return np.mean(accuracies)


        
hyperparameters = {}
hyperparameters['max_depth'] = [1,2,3,4,5,6]
hyperparameters['criteria_values'] = ["information_gain", "gini_index"]

best_accuracy = 0
best_hyperparameters = {}

out = {}
count = 0
for max_depth in hyperparameters['max_depth']:
        for criteria in hyperparameters['criteria_values']:
            # Create and fit the decision tree classifier with the current hyperparameters
            val_accuracy=k_fold(X,y,criteria=criteria,depth=max_depth)
            out[count] = {'max_depth': max_depth, 'criterion': criteria, 'val_accuracy': val_accuracy}
            count += 1
hparam_df = pd.DataFrame(out).T
print(hparam_df)
max_val=np.argmax(hparam_df['val_accuracy'])
opt_depth=hparam_df.at[max_val,'max_depth']
criteria_val=hparam_df.at[max_val,'criterion']
print(f"using 5-fold cross validation opt depth value for decision tree using criteria {criteria_val} is: ",opt_depth)