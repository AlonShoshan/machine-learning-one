# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:50:55 2017

@author: AlonA
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import copy

class clf2net:
    def __init__(self, clfList, clfNeural, useXtrain = False, random_state=1 , splitTrain = False):
        self.clfList = list(copy.deepcopy(clfList))
        self.clfNeural = copy.deepcopy(clfNeural)
        self.useXtrain = useXtrain # add the x_train to the input of the second stage
        self.random_state = random_state
        self.splitTrain = splitTrain
    
    def fit(self, X_train, y_train):
        X_train1, X_train2, y_train1, y_train2 = self.__splitTrain(X_train, y_train)
        y_pred_proba = np.zeros(shape=(len(X_train2),0))
        for clf in self.clfList:
            clf.fit(X_train1, y_train1)
            y_pred_proba_temp = clf.predict_proba(X_train2)
            y_pred_proba = np.concatenate((y_pred_proba, y_pred_proba_temp), axis=1)
        if self.useXtrain:
            y_pred_proba = pd.concat([X_train2,pd.DataFrame(data = y_pred_proba, index = X_train2.index.values)],axis=1)
        self.clfNeural.fit(y_pred_proba,y_train2)
        
    def predict(self,X_test):
        y_pred_proba = np.zeros(shape=(len(X_test),0))
        self.y_pred = []
        for clf in self.clfList:
            self.y_pred.append(clf.predict(X_test))
            y_pred_proba_temp = clf.predict_proba(X_test)
            y_pred_proba = np.concatenate((y_pred_proba, y_pred_proba_temp), axis=1)
        if self.useXtrain:
            y_pred_proba = pd.concat([X_test,pd.DataFrame(data = y_pred_proba, index = X_test.index.values)],axis=1)
        y_pred_neural = self.clfNeural.predict(y_pred_proba)
        self.y_pred.append(y_pred_neural)
        return y_pred_neural
    
    def predict_proba(self,X_test):
        y_pred_proba = np.zeros(shape=(len(X_test),0))
        for clf in self.clfList:
            y_pred_proba_temp = clf.predict_proba(X_test)
            y_pred_proba = np.concatenate((y_pred_proba, y_pred_proba_temp), axis=1)
        if self.useXtrain:
            y_pred_proba = pd.concat([X_test,pd.DataFrame(data = y_pred_proba, index = X_test.index.values)],axis=1)
        return self.clfNeural.predict_proba(y_pred_proba)
    
    def get_y_pred(self):
        return self.y_pred
        
    def accuracy_score(self,y_test):
        accuracy_score = []
        print("The accuracy score of the clf is:")
        for i in range(len(self.clfList)):
            res = metrics.accuracy_score(y_test,self.y_pred[i])
            accuracy_score.append(res)
            print(self.clfList[i], '##### The score is:', res) #TODO: add clf name
        res = metrics.accuracy_score(y_test,self.y_pred[i+1])
        accuracy_score.append(res)
        print(self.clfNeural, '##### The score is:', res) #TODO: add clf name
        return accuracy_score
    
    def classes_(self):
        return self.clfNeural.classes_
    
    def __splitTrain(self, X_train, y_train):
        if self.splitTrain:
             return train_test_split(X_train, pd.Series(y_train),
                                     test_size=0.5, random_state=self.random_state)
        return X_train, X_train, y_train, y_train
    
    def __repr__(self):
        return "clf2net(clfList=%s, clfNeural=%s, useXtrain=%s, random_state=%s, splitTrain=%s)" % (self.clfList, self.clfNeural, self.useXtrain, self.random_state, self.splitTrain)
        
    def __str__(self):
        return "clf2net(clfList=%s, clfNeural=%s, useXtrain=%s, random_state=%s, splitTrain=%s)" % (self.clfList, self.clfNeural, self.useXtrain, self.random_state, self.splitTrain)
        