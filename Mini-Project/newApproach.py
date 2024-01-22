import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from scipy import stats


def energy(a):
    n = len(a)
    total = 0
    for i in range(n):
        total+= a[i]**2

    total= total/n
    return total

classes = {"WALKING":1,"WALKING_UPSTAIRS":2,"WALKING_DOWNSTAIRS":3,"SITTING":4,"STANDING":5,"LAYING":6}

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Loading Training Data
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test= np.load('X_test.npy')
y_test = np.load('y_test.npy')
print("Successfully loaded Training Data...")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                                                # Training
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

### maybe we need to train with Total Acceleration, from paper....
X_train_accx = []
X_train_accy = []
X_train_accz = []


for i in range(len(X_train)):
    temp1 = []
    temp2 = []
    temp3 = []

    for j in range(len(X_train[0])):
        temp1.append(X_train[i][j][0])
        temp2.append(X_train[i][j][1])
        temp3.append(X_train[i][j][2])

    X_train_accx.append(temp1)
    X_train_accy.append(temp2)
    X_train_accz.append(temp3)

X_final_train=[]
X_final_train_mean= []
Y_final_train_mean= []
Z_final_train_mean= []
X_final_train_std= []
Y_final_train_std= []
Z_final_train_std= []
X_final_train_mad= []
Y_final_train_mad= []
Z_final_train_mad= []
X_final_train_max= []
Y_final_train_max= []
Z_final_train_max= []
X_final_train_min= []
Y_final_train_min= []
Z_final_train_min= []
X_final_train_energy= []
Y_final_train_energy= []
Z_final_train_energy= []

for i in range(len(X_train_accx)):
    X_final_train_mean.append(np.mean(X_train_accx[i]))
    Y_final_train_mean.append(np.mean(X_train_accy[i]))
    Z_final_train_mean.append(np.mean(X_train_accz[i]))
    X_final_train_std.append(np.std(X_train_accx[i]))
    Y_final_train_std.append(np.std(X_train_accy[i]))
    Z_final_train_std.append(np.std(X_train_accz[i]))
    X_final_train_mad.append(stats.median_abs_deviation(X_train_accx[i]))
    Y_final_train_mad.append(stats.median_abs_deviation(X_train_accy[i]))
    Z_final_train_mad.append(stats.median_abs_deviation(X_train_accz[i]))
    X_final_train_max.append(np.max(X_train_accx[i]))
    Y_final_train_max.append(np.max(X_train_accy[i]))
    Z_final_train_max.append(np.max(X_train_accz[i]))
    X_final_train_min.append(np.min(X_train_accx[i]))
    Y_final_train_min.append(np.min(X_train_accy[i]))
    Z_final_train_min.append(np.min(X_train_accz[i]))
    X_final_train_energy.append(energy(X_train_accx[i]))
    Y_final_train_energy.append(energy(X_train_accy[i]))
    Z_final_train_energy.append(energy(X_train_accz[i]))


X_final_train.append(X_final_train_mean)
X_final_train.append(Y_final_train_mean)
X_final_train.append(Z_final_train_mean)
X_final_train.append(X_final_train_std)
X_final_train.append(Y_final_train_std)
X_final_train.append(Z_final_train_std)
X_final_train.append(X_final_train_max)
X_final_train.append(Y_final_train_max)
X_final_train.append(Z_final_train_max)
X_final_train.append(X_final_train_min)
X_final_train.append(Y_final_train_min)
X_final_train.append(Z_final_train_min)
X_final_train.append(X_final_train_energy)
X_final_train.append(Y_final_train_energy)
X_final_train.append(Z_final_train_energy)
X_final_train.append(X_final_train_mad)
X_final_train.append(Y_final_train_mad)
X_final_train.append(Z_final_train_mad)

X_final_train = np.array(X_final_train)
X_final_train=np.transpose(X_final_train)

X_test_accx = []
X_test_accy = []
X_test_accz = []


for i in range(len(X_test)):
    temp1 = []
    temp2 = []
    temp3 = []

    for j in range(len(X_test[0])):
        temp1.append(X_test[i][j][0])
        temp2.append(X_test[i][j][1])
        temp3.append(X_test[i][j][2])

    X_test_accx.append(temp1)
    X_test_accy.append(temp2)
    X_test_accz.append(temp3)

X_final_test=[]
X_final_test_mean= []
Y_final_test_mean= []
Z_final_test_mean= []
X_final_test_std= []
Y_final_test_std= []
Z_final_test_std= []
X_final_test_mad= []
Y_final_test_mad= []
Z_final_test_mad= []
X_final_test_max= []
Y_final_test_max= []
Z_final_test_max= []
X_final_test_min= []
Y_final_test_min= []
Z_final_test_min= []
X_final_test_energy= []
Y_final_test_energy= []
Z_final_test_energy= []

for i in range(len(X_test_accx)):
    X_final_test_mean.append(np.mean(X_test_accx[i]))
    Y_final_test_mean.append(np.mean(X_test_accy[i]))
    Z_final_test_mean.append(np.mean(X_test_accz[i]))
    X_final_test_std.append(np.std(X_test_accx[i]))
    Y_final_test_std.append(np.std(X_test_accy[i]))
    Z_final_test_std.append(np.std(X_test_accz[i]))
    X_final_test_mad.append(stats.median_abs_deviation(X_test_accx[i]))
    Y_final_test_mad.append(stats.median_abs_deviation(X_test_accy[i]))
    Z_final_test_mad.append(stats.median_abs_deviation(X_test_accz[i]))
    X_final_test_max.append(np.max(X_test_accx[i]))
    Y_final_test_max.append(np.max(X_test_accy[i]))
    Z_final_test_max.append(np.max(X_test_accz[i]))
    X_final_test_min.append(np.min(X_test_accx[i]))
    Y_final_test_min.append(np.min(X_test_accy[i]))
    Z_final_test_min.append(np.min(X_test_accz[i]))
    X_final_test_energy.append(energy(X_test_accx[i]))
    Y_final_test_energy.append(energy(X_test_accy[i]))
    Z_final_test_energy.append(energy(X_test_accz[i]))


X_final_test.append(X_final_test_mean)
X_final_test.append(Y_final_test_mean)
X_final_test.append(Z_final_test_mean)
X_final_test.append(X_final_test_std)
X_final_test.append(Y_final_test_std)
X_final_test.append(Z_final_test_std)
X_final_test.append(X_final_test_max)
X_final_test.append(Y_final_test_max)
X_final_test.append(Z_final_test_max)
X_final_test.append(X_final_test_min)
X_final_test.append(Y_final_test_min)
X_final_test.append(Z_final_test_min)
X_final_test.append(X_final_test_energy)
X_final_test.append(Y_final_test_energy)
X_final_test.append(Z_final_test_energy)
X_final_test.append(X_final_test_mad)
X_final_test.append(Y_final_test_mad)
X_final_test.append(Z_final_test_mad)

X_final_test = np.array(X_final_test)
X_final_test=np.transpose(X_final_test)

print(X_final_test.shape)

# print(X_final_mean)


        

Recognizer = tree.DecisionTreeClassifier()
Recognizer = Recognizer.fit(X_final_train, y_train)
print("Descision Tree Trained Successfully!")

# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                                                 # Testing
# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# X_test = np.load('X_test.npy')
# y_test = np.load('y_test.npy')

# print("Successfully loaded Test Data...")

# X_test_total_acceleration = []
# for i in range(len(X_test)):
#     temp = []
#     for j in range(len(X_test[0])):
#         temp.append(np.dot(X_test[i][j],np.transpose(X_test[i][j])))
#     X_test_total_acceleration.append(temp)

# X_test_total_acceleration = np.array(X_test_total_acceleration)

y_pred = Recognizer.predict(X_final_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the Decision Tree model is (max_depth == None): ",accuracy)
con_mat = confusion_matrix(y_test,y_pred, labels=Recognizer.classes_)
print("Displaying Confusion Matrix...")
disp = ConfusionMatrixDisplay(confusion_matrix=con_mat,display_labels=Recognizer.classes_)
disp.plot()
plt.show()

# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                                                 # Tuning
# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# acc_vals = []
# acc_vals_ontrain = []
# con_mats = []
# depths = [i for i in range(2,9)]
# for depth in depths:
#     Recognizer = tree.DecisionTreeClassifier(max_depth=depth)
#     Recognizer = Recognizer.fit(X_train_total_acceleration, y_train)
#     y_pred = Recognizer.predict(X_test_total_acceleration)
#     y_pred_ontrain = Recognizer.predict(X_train_total_acceleration)
#     acc_vals.append(accuracy_score(y_test,y_pred))
#     acc_vals_ontrain.append(accuracy_score(y_train,y_pred_ontrain))
#     con_mats.append(confusion_matrix(y_test,y_pred, labels=Recognizer.classes_))

# ### Plotting the data
# num_plots = len(depths)

# # Create subplots
# fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))

# # Plot confusion matrices
# for i in range(num_plots):
#     axs[i//4, i%4].imshow(con_mats[i], cmap='Blues', interpolation='nearest')
#     axs[i//4, i%4].set_title(f"Max Depth = {depths[i]}")
#     axs[i//4, i%4].set_xlabel('Predicted')
#     axs[i//4, i%4].set_ylabel('True')

# # Plot accuracy values
# axs[1, 3].plot(depths, acc_vals, color='orange')
# axs[1, 3].plot(depths, acc_vals_ontrain, color='blue')
# axs[1, 3].set_title('Accuracy vs. Depth')
# axs[1, 3].set_xlabel('Depth -->')
# axs[1, 3].set_ylabel('Accuracy -->')
# axs[1, 3].legend()
# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()