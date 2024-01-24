import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from metrics import *


np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Q2 (A)
for criteria in ["gini", "entropy"]:
    tree = DecisionTreeClassifier(criterion=criteria)
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    print("Criteria:", criteria)
    print("Accuracy:", accuracy_score(y_test, y_hat))
    for cls in np.unique(y_test):
        print(f"Class: {cls}")
        print("Precision:", precision(y_test, y_hat, pos_label=cls))
        print("Recall:", recall(y_test, y_hat, pos_label=cls))

# Q2 (B)
def k_fold(X, y, criteria, depth, k=5):
    predictions = {}
    accuracies = []

    fold_size = len(X) // k

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]

        training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
        training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)

        dt_classifier = DecisionTreeClassifier(criterion=criteria, max_depth=depth)
        dt_classifier.fit(training_set, training_labels)

        fold_predictions = dt_classifier.predict(test_set)

        fold_accuracy = accuracy_score(test_labels, fold_predictions)

        predictions[i] = fold_predictions
        accuracies.append(fold_accuracy)

    return np.mean(accuracies)

hyperparameters = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'criteria_values': ['gini', 'entropy']}

best_accuracy = 0
best_hyperparameters = {}

out = {}
count = 0
for max_depth in hyperparameters['max_depth']:
    for criteria in hyperparameters['criteria_values']:
        val_accuracy = k_fold(X, y, criteria=criteria, depth=max_depth)
        out[count] = {'max_depth': max_depth, 'criterion': criteria, 'val_accuracy': val_accuracy}
        count += 1

hparam_df = pd.DataFrame(out).T
print(hparam_df)
