"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # print(y.dtype)
    if(y.dtype=='float64'):
        return True
    else:
        return False

    pass

def mean_square_col(Y:pd.Series):
    if(len(Y)==0):
        return 0
    ans=0
    mean=Y.mean()
    return (Y.apply(lambda x:(x-mean)**2).sum())/len(Y)

def mse(Y:pd.Series,attr:pd.Series) -> float:
    result=0
    total_val=len(attr)
    for attribute,val in attr.value_counts().items():
        result+=(val/total_val)*(mean_square_col(Y[attr==attribute]))
        
    return result

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    return Y.value_counts(normalize=True).apply(lambda x: -x*np.log2(x+1e-6)).sum()
    
    pass


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    return 1-Y.value_counts(normalize=True).apply(lambda x: x**2).sum()
    pass

def gini_gain(Y: pd.Series, attr: pd.Series,check_rin=False) -> float:
    """
    Function to calculate the information gain
    """        
    # print(mean_square_col(Y)-mse(Y,attr))
    # if(check_ifreal(Y)):
        
    #     return mean_square_col(Y)-mse(Y,attr)
    if(check_rin):
        attr = pd.Series(attr, name='Column1')
        Y_new = pd.Series(Y, name='Column2')
        df = pd.DataFrame({'Column1': attr, 'Column2': Y_new})
        df2 = df.sort_values(by='Column1').reset_index(drop=True)
        entropy_y=gini_index(Y)

        max_info =0
        chosen_split = 1e6
        for i in range(len(df2) - 1):
            ind1 = i
            ind2 = i + 1
            avg_val = (df2['Column1'].iloc[ind1] + df2['Column1'].iloc[ind2]) / 2
            Y1 = Y[:ind2]
            Y2 = Y[ind2:]
            en1 = gini_index(Y1)
            en2 = gini_index(Y2)
            net_en = entropy_y - ((len(Y1) / (len(Y1) + len(Y2))) * en1) - ((len(Y2) / (len(Y1) + len(Y2))) * en2)
            if net_en > max_info:
                max_info = net_en
                chosen_split = avg_val
        return max_info, chosen_split
    else:
        entropy_y=gini_index(Y)
        info_gain=entropy_y
        total_vals=len(attr)
        for attribute,val in attr.value_counts().items():
            info_gain-=val/total_vals*(gini_index(Y[attr == attribute]))
        return info_gain

def information_gain(Y: pd.Series, attr: pd.Series,check_rin=False) -> float:
    """
    Function to calculate the information gain
    """        
    # print(mean_square_col(Y)-mse(Y,attr))
    # if(check_ifreal(Y)):
        
    #     return mean_square_col(Y)-mse(Y,attr)
    if(check_rin):
        attr = pd.Series(attr, name='Column1')
        Y_new = pd.Series(Y, name='Column2')
        df = pd.DataFrame({'Column1': attr, 'Column2': Y_new})
        df2 = df.sort_values(by='Column1').reset_index(drop=True)
        entropy_y=entropy(Y)

        max_info =0
        chosen_split = 1e6
        for i in range(len(df2) - 1):
            ind1 = i
            ind2 = i + 1
            avg_val = (df2['Column1'].iloc[ind1] + df2['Column1'].iloc[ind2]) / 2
            Y1 = Y[:ind2]
            Y2 = Y[ind2:]
            en1 = entropy(Y1)
            en2 = entropy(Y2)
            net_en = entropy_y - ((len(Y1) / (len(Y1) + len(Y2))) * en1) - ((len(Y2) / (len(Y1) + len(Y2))) * en2)
            if net_en > max_info:
                max_info = net_en
                chosen_split = avg_val
        return max_info, chosen_split
    
    else:
        entropy_y=entropy(Y)
        info_gain=entropy_y
        total_vals=len(attr)
        for attribute,val in attr.value_counts().items():
            info_gain-=val/total_vals*(entropy(Y[attr == attribute]))
        return info_gain 
def info_for_real(Y: pd.Series, attr: pd.Series):
    attr = pd.Series(attr, name='Column1')
    Y_new = pd.Series(Y, name='Column2')
    df = pd.DataFrame({'Column1': attr, 'Column2': Y_new})
    df2 = df.sort_values(by='Column1').reset_index(drop=True)

    min_mse = 1e6
    min_mse_val = 0
    for i in range(len(df2) - 1):
        ind1 = i
        ind2 = i + 1
        avg_val = (df2['Column1'].iloc[ind1] + df2['Column1'].iloc[ind2]) / 2
        Y1 = Y[:ind2]
        Y2 = Y[ind2:]
        mse1 = mean_square_col(Y1)
        mse2 = mean_square_col(Y2)
        net_mse = (len(Y1) / (len(Y1) + len(Y2))) * mse1 + (len(Y2) / (len(Y1) + len(Y2))) * mse2
        if net_mse < min_mse:
            min_mse = net_mse
            min_mse_val = avg_val
    return min_mse, min_mse_val
def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series,check_rin=False):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.
    features: pd.Series is a list of all the attributes we have to split upon
    return: attribute to split upon
    """
    if(check_ifreal(y)):
        if(check_rin):
            if(criterion=="information_gain"):
                min_error=1e6
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(info_for_real(y,X[attribute])[0]<min_error):
                        min_error=info_for_real(y,X[attribute])[0]
                        chosen_split=info_for_real(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split
        
        
            else:
                min_error=1e6
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(info_for_real(y,X[attribute])[0]<min_error):
                        min_error=info_for_real(y,X[attribute])[0]
                        chosen_split=info_for_real(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split

        else:
            if(criterion=="information_gain"):
                
                # print(features)
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(information_gain(y,X[attribute])>max_gain):
                        max_gain=information_gain(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

            else:
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(information_gain(y,X[attribute])>max_gain):
                        max_gain=information_gain(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

 
    else:
        if(check_rin):
            if(criterion=="information_gain"):
                max_gain=0
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(information_gain(y,X[attribute],check_rin=True)[0]>max_gain):
                        max_gain=information_gain(y,X[attribute],check_rin=True)[0]
                        chosen_split=information_gain(y,X[attribute],check_rin=True)[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split
        
        
            else:
                max_gain=0
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(gini_gain(y,X[attribute],check_rin=True)[0]>max_gain):
                        max_gain=gini_gain(y,X[attribute],check_rin=True)[0]
                        chosen_split=gini_gain(y,X[attribute],check_rin=True)[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split

              
        else:
            if(criterion=="information_gain"):
                
                # print(features)
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(information_gain(y,X[attribute])>max_gain):
                        max_gain=information_gain(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

            else:
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(gini_gain(y,X[attribute])>max_gain):
                        max_gain=gini_gain(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    X_match, y_match = X[X[attribute] == value], y[X[attribute] == value]
    return (X_match,y_match)
    pass
