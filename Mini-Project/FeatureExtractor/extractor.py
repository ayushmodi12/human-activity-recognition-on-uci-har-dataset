import numpy as np
import tsfel

def energy(a):
    return (np.sum(a**2))/len(a)

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

    X_final = []
    for i in range(len(X)):
        feature = []
        
        feature.append(np.mean(X[i][:][0]))
        feature.append(np.mean(X[i][:][1]))
        feature.append(np.mean(X[i][:][2]))
        
        feature.append(np.std(X[i][:][0]))
        feature.append(np.std(X[i][:][1]))
        feature.append(np.std(X[i][:][2]))
        
        feature.append(np.max(X[i][:][0]))
        feature.append(np.max(X[i][:][1]))
        feature.append(np.max(X[i][:][2]))

        feature.append(np.min(X[i][:][0]))
        feature.append(np.min(X[i][:][1]))
        feature.append(np.min(X[i][:][2]))

        feature.append(energy(X[i][:][0]))
        feature.append(energy(X[i][:][1]))
        feature.append(energy(X[i][:][2]))
        
        feature.append(tsfel.median_abs_deviation(X[i][:][0]))
        feature.append(tsfel.median_abs_deviation(X[i][:][1]))
        feature.append(tsfel.median_abs_deviation(X[i][:][2]))
        
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
    
    return np.array(X_final)
