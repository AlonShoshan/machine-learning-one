# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:16:36 2017

@author: Alon
"""

import numpy as np
import pandas as pd
from sklearn import metrics

class clfNdfOtherAll:
    def __init__(self, clf1th, clf2nd):
        self.clf1th = clf1th
        self.clf2nd = clf2nd
        
    def fit(self, X_train, y_train):
        X_train['y_label'] = y_train
        X_train_NdfOther, y_train_NdfOther = self.sortNdfOther(X_train)
        X_train_non_Ndf, y_train_non_Ndf   = self.sortNonNdf(X_train)
        self.clf1th.fit(X_train_NdfOther, y_train_NdfOther)
        self.clf2nd.fit(X_train_non_Ndf, y_train_non_Ndf)
        
    def predict(self,X_test_org):
        X_test = X_test_org.copy(deep=True)
        Indexs = range(len(X_test.index.values))
        X_test['new_indexes'] = Indexs
        X_test = X_test.set_index('new_indexes')
        X_test.index.name = None
        y_pred_NdfOther = self.clf1th.predict(X_test)
        y_pred_NdfOther = pd.DataFrame(data=y_pred_NdfOther,index=X_test.index.values, columns = ['y_label'])
        X_test_nonNdf = self.sortNonNdfFromTest(y_pred_NdfOther, X_test)
        y_pred_nonNdf = self.clf2nd.predict(X_test_nonNdf)
        y_pred_nonNdf = pd.DataFrame(data=y_pred_nonNdf,index=X_test_nonNdf.index.values, columns = ['y_label'])
        y_pred = self.meargeYpred(y_pred_NdfOther,y_pred_nonNdf,X_test)
        self.clf1thPred = y_pred_NdfOther
        self.clf2ndPred = y_pred_nonNdf
        self.y_pred = y_pred
        return y_pred
        
    def sortNdfOther(self, X_train):
        X_train_NdfOther = X_train.copy(deep=True)
        X_train_NdfOther.loc[(X_train_NdfOther.y_label == 'NDF'), 'y_label'] = 'NDF'
        X_train_NdfOther.loc[(X_train_NdfOther.y_label != 'NDF'), 'y_label'] = 'other'
        y_train_NdfOther = X_train_NdfOther['y_label']
        return X_train_NdfOther.drop(['y_label'], 1), y_train_NdfOther
        
    def sortNonNdf(self, X_train):
        X_train_temp = X_train.copy(deep=True)
        X_train_non_Ndf = X_train_temp[X_train_temp.y_label != 'NDF']
        y_train_non_Ndf = X_train_non_Ndf['y_label']
        return X_train_non_Ndf.drop(['y_label'], 1), y_train_non_Ndf
        
    def sortNonNdfFromTest(self, y_pred_NdfOther, X_test):
        X_test_temp = X_test.copy(deep=True)
        y_pred_NdfOtherDF_onlyNDF = y_pred_NdfOther[y_pred_NdfOther.y_label == 'NDF']
        return X_test_temp.drop(y_pred_NdfOtherDF_onlyNDF.index.values)
        
    def meargeYpred(self, y_pred_NdfOther, y_pred_nonNdf, X_test):
        y_pred_NdfOther_onlyNdf = y_pred_NdfOther[y_pred_NdfOther.y_label == 'NDF']
        y_pred = list(range(len(X_test.index.values)))
        for Index in y_pred_NdfOther_onlyNdf.index.values:
            y_pred[Index] = y_pred_NdfOther_onlyNdf['y_label'][Index]
        for Index in y_pred_nonNdf.index.values:
            y_pred[Index] = y_pred_nonNdf['y_label'][Index]
        return y_pred
        
    def accuracy_score(self, y_test):
        print(y_test)
        y_test_temp = pd.DataFrame(data=list(y_test), columns = ['y_label'])
        y_test_NdfOther = y_test_temp.copy(deep=True)
        y_test_NdfOther.loc[(y_test_NdfOther.y_label == 'NDF'), 'y_label'] = 'NDF'
        y_test_NdfOther.loc[(y_test_NdfOther.y_label != 'NDF'), 'y_label'] = 'other'
        y_test_non_Ndf = y_test_temp[y_test_temp.y_label != 'NDF']        
        
        accuracy_score = []
        
        print('\n\n\n1.1:',y_test_NdfOther)
        print('1.2:',pd.DataFrame(data=self.clf1thPred,columns = ['y_label']))
        print('2.1"',y_test_non_Ndf)
        print('2.2:',pd.DataFrame(data=self.clf2ndPred,columns = ['y_label']))
        print('3.1:',y_test)
        print('3.2',pd.DataFrame(data=self.y_pred,columns = ['y_label']))
        accuracy_score.append(metrics.accuracy_score(list(y_test_NdfOther), list(self.clf1thPred)))
        accuracy_score.append(metrics.accuracy_score(list(y_test_non_Ndf), list(self.clf2ndPred)))
        accuracy_score.append(metrics.accuracy_score(list(y_test), list(self.y_pred)))
        return accuracy_score
    
        
        
        
    