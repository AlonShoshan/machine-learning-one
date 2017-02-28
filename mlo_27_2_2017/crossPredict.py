# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:14:49 2016

@author: Alon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import *
from pylab import *
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import original_functions.originalFunctions3 as orig

fitPredictAndTestClass = orig.fitPredictAndTestClass()

users = orig.loadAndUpdateAirBnbUsers('input/train_users.csv',13,95,addValidAgeBit=True)
festureList = [ 'affiliate_channel','affiliate_provider','first_affiliate_tracked','first_browser','first_device_type','language',
                'signup_app','signup_method','signup_flow','date_account_created_month','date_first_active_month','date_account_created_dayofweek',
                'date_first_active_dayofweek','age','validAge','gender' ]
predictMethod = tree.DecisionTreeClassifier()
category = 'country_destination'

fitPredictAndTestClass.trainTestSplit(users,festureList,category)

y_pred0 = fitPredictAndTestClass.PredictAndTest(predictMethod,crossPredict=True)
print('y_pred0',y_pred0['y_pred'])
print('---------------------------------------------------------')
print('accuracy_score0',y_pred0['accuracy_score'])
print('---------------------------------------------------------')
print('y_pred_log0[0:1]',y_pred0['y_pred_log'][0:1])
print('---------------------------------------------------------')
print('y_pred_proba0[0:1]',y_pred0['y_pred_proba'][0:1])
print('---------------------------------------------------------')
print('classes',predictMethod.classes_)

print('\n\n')

users = orig.loadAndUpdateAirBnbUsers('input/train_users.csv',13,95,addValidAgeBit=True)
predictMethod = LogisticRegression()
y_pred = fitPredictAndTestClass.PredictAndTest(predictMethod,crossPredict=True)
print('y_pred',y_pred['y_pred'])
print('---------------------------------------------------------')
print('accuracy_score',y_pred['accuracy_score'])
print('---------------------------------------------------------')
print('y_pred_log[0:1]',y_pred['y_pred_log'][0:1])
print('---------------------------------------------------------')
print('y_pred_proba[0:1]',y_pred['y_pred_proba'][0:1])
print('---------------------------------------------------------')
print('classes',predictMethod.classes_)

#print('\n\n')
#print('---------------------------------------------------------')
#print('y_pred_log[0:1]+',y_pred_log[0:1]+y_pred_log0[0:1])
#print('---------------------------------------------------------')
#print('y_pred_proba[0:1]+',y_pred_proba[0:1]+y_pred_proba0[0:1])
#print('---------------------------------------------------------')
#print('classes',predictMethod.classes_)

class crossPredictions:
    def __init__(self,usePrints=False):
        self.usePrints = usePrints
        self.user = orig.loadAndUpdateAirBnbUsers('input/train_users.csv',13,95,addValidAgeBit=True)
        self.festureList = [ 'affiliate_channel','affiliate_provider','first_affiliate_tracked','first_browser','first_device_type','language',
                'signup_app','signup_method','signup_flow','date_account_created_month','date_first_active_month','date_account_created_dayofweek',
                'date_first_active_dayofweek','age','validAge','gender' ]
        self.category = 'country_destination'
        self.logreg = LogisticRegression()
        self.DecisionTreeClassifier = tree.DecisionTreeClassifier()
        self.fitPredictAndTestClass = orig.fitPredictAndTestClass()
        self.fitPredictAndTestClass.trainTestSplit(self.user,self.festureList,self.category)
        
    def run(self):
        self.step()
        
    def step(self):
        if self.usePrints :
            print('run Step')
        self.logregYPredDictionary = self.fitPredictAndTestClass.PredictAndTest(self.logreg,crossPredict=True)
        self.DecisionTreeClassifierYPredDictionary = self.fitPredictAndTestClass.PredictAndTest(self.DecisionTreeClassifier,crossPredict=True)
        self.classes = self.logreg.classes_
        if self.classes != self.DecisionTreeClassifier.classes_ :
            print('not same classes',self.classes == self.DecisionTreeClassifier.classes_)
    
    def superPosition(self):
        if self.usePrints :
            print('run superPosition')
        self.superPositionPred = self.logregYPredDictionary['y_pred_proba'] + self.DecisionTreeClassifierYPredDictionary['y_pred_proba']
        self.new_pred = []
        for i in range(self.fitPredictAndTestClass.y_train):
            maxVal = 0
            maxIndex = 0
            for j in range(len(self.logreg.classes_)):
                if maxVal < self.fitPredictAndTestClass[i][j]:
                    maxVal = self.fitPredictAndTestClass[i][j]
                    maxIndex = j
            self.new_pred[i] = self.classes[maxIndex]
        self.accuracy_score = metrics.accuracy_score(self.fitPredictAndTestClass.y_test, self.new_pred)
        print(self.accuracy_score)
                
    
cl = crossPredictions(usePrints=True)   
cl.run()
cl.superPosition() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        