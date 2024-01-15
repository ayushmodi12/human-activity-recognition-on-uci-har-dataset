"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *
from collections import Counter
np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree=None
    def DIDO(self,X: pd.DataFrame, y: pd.Series,max_depth=8):
        attribute_names=list(X.columns)
        cnt = Counter(x for x in y)
        if len(cnt) == 1:
            return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
        ## Second check: Is this split of the dataset empty? if yes, return a default value
        elif len(X)==0 or (not attribute_names):
            return None

        else:               
          best_attr = opt_split_attribute(X,y,criterion=self.criterion,features=attribute_names)    
          # X.drop(best_attr, axis=1, inplace=True)
          tree = {best_attr:{}} # Initiate the tree with best attribute as a node
          if(len(X)==0):
            return None
          for attr_val, data_subset in X.groupby(by=best_attr,as_index=False):
            data_subset=data_subset.drop(best_attr,axis=1)
            y_new=y[X[best_attr]==attr_val]
            subtree=self.DIDO(data_subset,y_new)
            tree[best_attr][attr_val]=subtree
          self.tree=tree
          return tree
    def DIRO(self,X: pd.DataFrame, y: pd.Series,max_depth=8):
        attribute_names=list(X.columns)
        cnt = Counter(x for x in y)
        if len(cnt) == 1:
            return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
        ## Second check: Is this split of the dataset empty? if yes, return a default value
        elif len(X)==0 or (not attribute_names):
            return y.mean()

        else:               
          best_attr = opt_split_attribute(X,y,criterion=self.criterion,features=attribute_names)    
          # X.drop(best_attr, axis=1, inplace=True)
          tree = {best_attr:{}} # Initiate the tree with best attribute as a node
          if(len(X)==0):
            return None
          for attr_val, data_subset in X.groupby(by=best_attr,as_index=False):
            data_subset=data_subset.drop(best_attr,axis=1)
            y_new=y[X[best_attr]==attr_val]
            subtree=self.DIRO(data_subset,y_new)
            tree[best_attr][attr_val]=subtree
          self.tree=tree
          return tree
    def RIDO(self,X: pd.DataFrame, y: pd.Series,max_depth=8):
        attribute_names=list(X.columns)
        cnt = Counter(x for x in y)
        if len(cnt) == 1:
            return next(iter(cnt))  
        elif len(X)==0 or (not attribute_names):
            return None
        else:               
          best_attr = opt_split_attribute(X,y,criterion=self.criterion,features=attribute_names)    
          # X.drop(best_attr, axis=1, inplace=True)
          tree = {best_attr:{}} # Initiate the tree with best attribute as a node
          if(len(X)==0):
            return None
          for attr_val, data_subset in X.groupby(by=best_attr,as_index=False):
            data_subset=data_subset.drop(best_attr,axis=1)
            y_new=y[X[best_attr]==attr_val]
            subtree=self.DIFO(data_subset,y_new)
            tree[best_attr][attr_val]=subtree
          self.tree=tree
          return tree
    def RIRO(self,X:pd.DataFrame,y:pd.Series):
      attribute_names=list(X.columns)
      cnt = Counter(x for x in y)
      print(X)
      if len(cnt) == 1:
          return y.mean()
          ## Second check: Is this split of the dataset empty? if yes, return a default value
      elif len(X)==0 or (not attribute_names):
              return {}
      attribute,split_value=opt_split_attribute(X,y,criterion="information_gain",features=pd.Series(list(X.columns)),check_rin=True)

      #X['out']=y.copy()
      # print(X)
      X_new=X
      X_new.loc[:, 'out'] = y.copy()
      X_new=X_new.sort_values(by=attribute,ascending=True)
      X_new=X_new.reset_index()
      X_new=X_new.drop(['index'],axis=1)
      # y_new=X['out']
      # X.drop(['out'],axis=1)
      print(attribute)
      tree={attribute:{}}
      
      data_subset_less=pd.DataFrame(X_new[X_new[attribute]<=split_value])
      y_less=pd.Series(data_subset_less['out'])
      df2=data_subset_less.drop(['out'],axis=1)
      print(df2)
      df3=df2.drop(attribute,axis=1)
      # print(df3)
      # return 
      split1="Less than " + str(split_value)
      subtree_less=self.RIRO(df3,y_less)
      
      tree[attribute][split1] =subtree_less 
      
      data_subset_more=X_new[X_new[attribute]>split_value]
      y_more=pd.Series(data_subset_more['out'])
      df4=data_subset_more.drop(['out'],axis=1)
      df5=df4.drop(attribute,axis=1)
      split2="Greater than " + str(split_value)
      subtree_more=self.RIRO(df5,y_more)
      tree[attribute][split2]=subtree_more
      
      return tree
    def RIDO(self,X:pd.DataFrame,y:pd.Series):
      attribute_names=list(X.columns)
      cnt = Counter(x for x in y)
      print(X)
      if len(cnt) == 1:
          return next(iter(cnt))
          ## Second check: Is this split of the dataset empty? if yes, return a default value
      elif len(X)==0 or (not attribute_names):
              return None
      
      attribute,split_value=opt_split_attribute(X,y,criterion="information_gain",features=pd.Series(list(X.columns)),check_rin=True)
      
      X_new=X
      X_new.loc[:, 'out'] = y.copy()
      X_new=X_new.sort_values(by=attribute,ascending=True)
      X_new=X_new.reset_index()
      X_new=X_new.drop(['index'],axis=1)
      tree={attribute:{}}
      
      data_subset_less=pd.DataFrame(X_new[X_new[attribute]<=split_value])
      y_less=pd.Series(data_subset_less['out'])
      df2=data_subset_less.drop(['out'],axis=1)
      # print(df2)
      # df3=df2.drop(attribute,axis=1)
      # print(df3)
      # return 
      split1="Less than " + str(split_value)
      subtree_less=self.RIDO(df2,y_less)
      
      tree[attribute][split1] =subtree_less 
      
      data_subset_more=X_new[X_new[attribute]>split_value]
      y_more=pd.Series(data_subset_more['out'])
      df4=data_subset_more.drop(['out'],axis=1)
      # df5=df4.drop(attribute,axis=1)
      split2="Greater than " + str(split_value)
      subtree_more=self.RIDO(df4,y_more)
      tree[attribute][split2]=subtree_more
      
      return tree
    def fit(self, X: pd.DataFrame, y: pd.Series):
        if(check_ifreal(y)):
            if(check_ifreal(X[0])):
              return self.RIRO(X,y)
            else:
              return self.DIRO(X,y)
        else:
            if(check_ifreal(X[0])):
              return self.RIDO(X,y)
            else:
              return self.DIDO(X,y)
           
    def predict_help(self,row,tree):
        if(tree!=None):
          attribute=next(iter(tree))
          if row[attribute] in tree[attribute].keys():
              result=tree[attribute][row[attribute]]
              if isinstance(result,dict):
                  return self.predict_help(row,result)
              else:
                  return result
        else:
           return 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        list=[]
        output=""
        
        for row in X.iloc:
            tree=self.tree
            list.append(self.predict_help(row,tree))
        return pd.Series(list)
            
        # Traverse the tree you constructed to return the predicted values for the given test inputs.

        pass

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pass
