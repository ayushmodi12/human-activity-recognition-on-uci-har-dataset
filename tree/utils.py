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
    if(y.dtype=='float64'):
        return True
    else:
        return False
    pass

def mean_square_col(Y:pd.Series):
    if(len(Y)==0):
        return 0
    mean=Y.mean()
    return (Y.apply(lambda x:(x-mean)**2).sum())/len(Y)

# def mse(Y:pd.Series,attr:pd.Series) -> float:
#     result=0
#     total_val=len(attr)
#     for attribute,val in attr.value_counts().items():
#         result+=(val/total_val)*(mean_square_col(Y[attr==attribute]))
        
#     return result

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    return Y.value_counts(normalize=True).apply(lambda x: -x*np.log2(x+1e-6)).sum()



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    return 1-Y.value_counts(normalize=True).apply(lambda x: x**2).sum()

def gini_gain(Y: pd.Series, attr: pd.Series) -> float:
        entropy_y=gini_index(Y)
        info_gain=entropy_y
        total_vals=len(attr)
        for attribute,val in attr.value_counts().items():
            info_gain-=val/total_vals*(gini_index(Y[attr == attribute]))
        return info_gain 

def info_dido(Y: pd.Series, attr: pd.Series):
        entropy_y=entropy(Y)
        info_gain=entropy_y
        total_vals=len(attr)
        for attribute,val in attr.value_counts().items():
            info_gain-=val/total_vals*(entropy(Y[attr == attribute]))
        return info_gain 
    
def info_diro(Y: pd.Series, attr: pd.Series):
        sample_mse=mean_square_col(Y)
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        total_mse=0
        for value in attr.unique():
            total_mse += mean_square_col(df_samples[attr==value]['Y'])*(len(df_samples[attr==value]['Y'])/len(Y))
        info_gain = sample_mse -total_mse
        return info_gain
    
def info_riro(Y: pd.Series, attr: pd.Series):
    split_val = -1e6
    max_gain = -1e6
    df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
    df_samples = df_samples.sort_values('Attribute')
    attributes = attr.sort_values()
    lower_ind = attributes.index[0]

    for upper_ind in attributes.index[1:]:
        avg = (attributes[lower_ind] + attributes[upper_ind])/2
        gain = mean_square_col(Y) - mean_square_col(df_samples.loc[attr<=avg]['Y']) * (len(df_samples.loc[attr<=avg]['Y'])/len(Y)) - mean_square_col(df_samples.loc[attr>avg]['Y']) * (len(df_samples.loc[attr>avg]['Y'])/len(Y))
        lower_ind=upper_ind
        if gain>max_gain:
            split_val=avg
            max_gain=gain
    return max_gain,split_val

def info_rido(Y: pd.Series, attr: pd.Series):
        split_val = -1e6
        max_gain = -1e6
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        df_samples = df_samples.sort_values('Attribute')
        attributes = attr.sort_values()
        lower_ind = attributes.index[0]
        for upper_ind in attributes.index[1:]:
            avg = (attributes[lower_ind] + attributes[upper_ind])/2
            gain = entropy(Y) - entropy(df_samples.loc[attr<=avg]['Y']) * (len(df_samples.loc[attr<=avg]['Y'])/len(Y)) - entropy(df_samples.loc[attr>avg]['Y']) * (len(df_samples.loc[attr>avg]['Y'])/len(Y))
            if gain>max_gain:
                split_val=avg
                max_gain=gain
            lower_ind=upper_ind
        # print(max_gain)
        return max_gain,split_val

def gini_rido(Y: pd.Series, attr: pd.Series):
        split_val = -1e6
        max_gain = -1e6
        df_samples = pd.DataFrame({'Attribute':attr,'Y':Y})
        df_samples = df_samples.sort_values('Attribute')  
        attributes = attr.sort_values()
        lower_ind = attributes.index[0]
        for upper_ind in attributes.index[1:]:
            avg = (attributes[lower_ind] + attributes[upper_ind])/2
            # print(attr)
            gain = gini_index(Y) - gini_index(df_samples.loc[attr<=avg]['Y']) * (len(df_samples.loc[attr<=avg]['Y'])/len(Y)) - gini_index(df_samples.loc[attr>avg]['Y']) * (len(df_samples.loc[attr>avg]['Y'])/len(Y))
            if gain>max_gain:
                split_val=avg
                max_gain=gain
            lower_ind=upper_ind
        return max_gain,split_val

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
            #RIRO
            if(criterion=="information_gain"):
                min_error=-1e6
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(info_riro(y,X[attribute])[0]>min_error):
                        min_error=info_riro(y,X[attribute])[0]
                        chosen_split=info_riro(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split
        
        
            else:
                min_error=1e6
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:

                    if(info_riro(y,X[attribute])[0]<min_error):
                        min_error=info_riro(y,X[attribute])[0]
                        chosen_split=info_riro(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split

        else:
            #DIRO
            if(criterion=="information_gain"):
                
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(info_diro(y,X[attribute])>max_gain):
                        max_gain=info_diro(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

            else:
                max_gain=0
                chosen_attribute=""
                for attribute in features:
                    if(info_diro(y,X[attribute])>max_gain):
                        max_gain=info_diro(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

 
    else:
        if(check_rin):
            #RIDO
            if(criterion=="information_gain"):
                max_gain=-1
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(info_rido(y,X[attribute])[0]>=max_gain):
                        max_gain=info_rido(y,X[attribute])[0]
                        chosen_split=info_rido(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split
        
        
            else:
                max_gain=-1
                chosen_attribute=""
                chosen_split=1e6
                for attribute in features:
                    if(gini_rido(y,X[attribute])[0]>max_gain):
                        max_gain=gini_rido(y,X[attribute])[0]
                        chosen_split=gini_rido(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split

              
        else:
            #DIDO
            if(criterion=="information_gain"):
                max_gain=-1
                chosen_attribute=""
                for attribute in features:
                    if(info_dido(y,X[attribute])>max_gain):
                        max_gain=info_dido(y,X[attribute])
                        chosen_attribute=attribute
                return chosen_attribute

            else:
                max_gain=-1
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
    X_match.drop([attribute], axis=1)
    return (X_match,y_match)
    pass
