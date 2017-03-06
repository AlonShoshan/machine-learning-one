# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:50:55 2017

@author: AlonA
"""
import numpy as np
from sklearn import metrics

class clf2net:
    def __init__(self, clfList, clfNeural):
        self.clfList = clfList
        self.clfNeural = clfNeural
        
    def fit(self, X_train, y_train):
        y_pred_proba = np.zeros(shape=(len(X_train),0))
        for clf in self.clfList:
            clf.fit(X_train, y_train)
            y_pred_proba_temp = clf.predict_proba(X_train)
            y_pred_proba = np.concatenate((y_pred_proba, y_pred_proba_temp), axis=1)
        
        self.clfNeural.fit(y_pred_proba,y_train)
        return y_pred_proba
        
    def predict(self,X_test):
        y_pred_proba = np.zeros(shape=(len(X_test),0))
        self.y_pred = []
        for clf in self.clfList:
            self.y_pred.append(clf.predict(X_test))
            y_pred_proba_temp = clf.predict_proba(X_test)
            y_pred_proba = np.concatenate((y_pred_proba, y_pred_proba_temp), axis=1)
        y_pred_neural = self.clfNeural.predict(y_pred_proba)
        self.y_pred.append(y_pred_neural)
        return y_pred_neural, y_pred_proba
    
    
    def accuracy_score(self,y_test):
        accuracy_score = []
        for i in self.y_pred:
            res = metrics.accuracy_score(y_test, i)
            accuracy_score.append(res)
            print('pred:', res) #TODO: add clf name
        return accuracy_score