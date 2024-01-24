def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series,check_rin):
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
                    if(mse(y,X[attribute])[0]<min_error):
                        min_error=mse(y,X[attribute])[0]
                        chosen_split=mse(y,X[attribute])[1]
                        chosen_attribute=attribute
                return chosen_attribute,chosen_split
        
        
            else:
                min_gini_ind=0
                chosen_attribute=""
                for attribute in features:
                    if(gini_index(y)<min_gini_ind):
                        chosen_attribute=attribute
                        min_gini_ind=gini_index(y)
                return chosen_attribute
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
                min_gini_ind=0
                chosen_attribute=""
                for attribute in features:
                    if(gini_index(y)<min_gini_ind):
                        chosen_attribute=attribute
                        min_gini_ind=gini_index(y)
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
                min_gini_ind=0
                chosen_attribute=""
                for attribute in features:
                    if(gini_index(y)<min_gini_ind):
                        chosen_attribute=attribute
                        min_gini_ind=gini_index(y)
                return chosen_attribute
              
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
                min_gini_ind=0
                chosen_attribute=""
                for attribute in features:
                    if(gini_index(y)<min_gini_ind):
                        chosen_attribute=attribute
                        min_gini_ind=gini_index(y)
                return chosen_attribute

