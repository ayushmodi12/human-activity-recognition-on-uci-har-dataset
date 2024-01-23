import numpy as np
import tsfel
from scipy import stats

def energy(a):
    n = len(a)
    total = 0
    for i in range(n):
        total+= a[i]**2

    total= total/n
    return total

def extract(X):
    '''
    X is a the 3-d array --> input
    '''
    X_total = []
    for i in range(len(X)):
        temp = []
        for j in range(len(X[0])):
            temp.append(np.sqrt(np.dot(X[i][j],np.transpose(X[i][j]))))
        X_total.append(temp)
    X_train_accx = []
    X_train_accy = []
    X_train_accz = []


    for i in range(len(X)):
        temp1 = []
        temp2 = []
        temp3 = []

        for j in range(len(X[0])):
            temp1.append(X[i][j][0])
            temp2.append(X[i][j][1])
            temp3.append(X[i][j][2])

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
    X_final = []
    for i in range(len(X)):
        feature = []
        
        # feature.append(np.mean(X[i][:][0]))
        # feature.append(np.mean(X[i][:][1]))
        # feature.append(np.mean(X[i][:][2]))
        
        # feature.append(np.std(X[i][:][0]))
        # feature.append(np.std(X[i][:][1]))
        # feature.append(np.std(X[i][:][2]))
        
        # feature.append(tsfel.median_abs_deviation(X[i][:][0]))
        # feature.append(tsfel.median_abs_deviation(X[i][:][1]))
        # feature.append(tsfel.median_abs_deviation(X[i][:][2]))

        # feature.append(np.max(X[i][:][0]))
        # feature.append(np.max(X[i][:][1]))
        # feature.append(np.max(X[i][:][2]))

        # feature.append(np.min(X[i][:][0]))
        # feature.append(np.min(X[i][:][1]))
        # feature.append(np.min(X[i][:][2]))

        # feature.append(energy(X[i][:][0]))
        # feature.append(energy(X[i][:][1]))
        # feature.append(energy(X[i][:][2]))
        
        feature.append(tsfel.auc(X_total[i],50))
        
        feature.append(tsfel.interq_range(X[i][:][0]))
        feature.append(tsfel.interq_range(X[i][:][1]))
        feature.append(tsfel.interq_range(X[i][:][2]))
        
        feature.append(tsfel.entropy(X[i][:][0]))
        feature.append(tsfel.entropy(X[i][:][1]))
        feature.append(tsfel.entropy(X[i][:][2]))
        
        feature.append(np.corrcoef(X[i][:][0],X[i][:][1])[0][1])
        feature.append(np.corrcoef(X[i][:][1],X[i][:][2])[0][1])
        feature.append(np.corrcoef(X[i][:][0],X[i][:][2])[0][1])
        
        X_final.append(feature)
    X_train_temp = []
    for i in range(len(X_final_train)):
        X_train_temp.append(list(X_final_train[i]) + list(X_final[i]))
    X_train_temp = np.array(X_train_temp)
    return np.array(X_train_temp)
