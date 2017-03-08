# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:46:09 2016

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
import time

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
        self.startRun = time.clock() 
        self.step()
        self.superPosition()
        print('douration:', time.clock() - self.startRun)
        
    def step(self):
        if self.usePrints :
            print('run Step')
            print('logreg fit')
        self.logregYPredDictionary = self.fitPredictAndTestClass.PredictAndTest(self.logreg,crossPredict=True)
        if self.usePrints :
            print('logreg score:',self.logregYPredDictionary['accuracy_score'])
            print('DecisionTree fit')
        self.DecisionTreeClassifierYPredDictionary = self.fitPredictAndTestClass.PredictAndTest(self.DecisionTreeClassifier,crossPredict=True)
        if self.usePrints :
            print('DecisionTree score:',self.DecisionTreeClassifierYPredDictionary['accuracy_score'])
        self.classes = self.logreg.classes_
        if (self.classes != self.DecisionTreeClassifier.classes_).any() :
            print('logreg classes',self.logreg.classes_)
            print('DecisionTree classes',self.DecisionTreeClassifier.classes_)
            print('not same classes',self.classes == self.DecisionTreeClassifier.classes_)
    
    def superPosition(self):
        if self.usePrints :
            print('run superPosition')
        self.superPositionPred = self.logregYPredDictionary['y_pred_proba'] + 0.2*self.DecisionTreeClassifierYPredDictionary['y_pred_proba']
        self.new_pred = []
        for i in range(len(self.fitPredictAndTestClass.y_test)):
            maxVal = 0
            maxIndex = 0
            for j in range(len(self.classes)):
                if maxVal < self.superPositionPred[i][j]:
                    maxVal = self.superPositionPred[i][j]
                    maxIndex = j
            self.new_pred.append(self.classes[maxIndex])
        self.accuracy_score = metrics.accuracy_score(self.fitPredictAndTestClass.y_test, self.new_pred)
        print('logreg        score:',self.logregYPredDictionary['accuracy_score'])
        print('DecisionTree  score:',self.DecisionTreeClassifierYPredDictionary['accuracy_score'])
        print('superPosition score:',self.accuracy_score)
                
    
cl = crossPredictions(usePrints=True)   
cl.run()