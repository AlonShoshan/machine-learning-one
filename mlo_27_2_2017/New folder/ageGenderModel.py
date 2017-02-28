# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:26:04 2016

@author: Alon
"""
# virsion 1.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import *
from pylab import *
import random
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# ageGenderModel is a Clasifier
# inializizion by giving it 4 other prediction models
class ageGenderModel:
    def __init__(self, predictMethod0, predictMethod1, predictMethod2, predictMethod3, usePrints=False):
        self.predictMethod0 = predictMethod0
        self.predictMethod1 = predictMethod1
        self.predictMethod2 = predictMethod2
        self.predictMethod3 = predictMethod3
        self.usePrints = usePrints
        self.didPreProssesing = False
    
    # Sort users who have age and who dosent have age
    def sortAge(self, users):
        if ((users.columns == 'ageValidBit').any()): # if users have validAge bit
            X_nan_age = users[users.ageValidBit == 0].drop(['age', 'ageValidBit'], 1)
            X_age = users[users.ageValidBit == 1].drop('ageValidBit', 1)
        else: # if users have dosent validAge bit
            X_nan_age = users[users.age == '-unknown-'].drop('age',1)
            X_age = users[users.age != '-unknown-']
        return X_nan_age, X_age

    # Sort users who have gender and who dosent have gender
    def sortGender(self,users):
        X_nan_gender = users[users.genderValidBit == 1].drop(['genderValidBit'], 1)
        X_gender = users[users.genderValidBit == 0].drop('genderValidBit', 1)
        for i in range(5):
            if ((X_nan_gender.columns == ('gender' + str(i))).any()) :
                X_nan_gender = X_nan_gender.drop([('gender'+str(i))], 1)
        return X_nan_gender, X_gender

    # Sort users to 4 groups: yes/no gender/age
    def sortAgeGender(self, users, usePrints=False):
        X_nan_age, X_age = self.sortAge(users)
        X_nan_age_nan_gender, X_nan_age_gender = self.sortGender(X_nan_age)
        X_age_nan_gender, X_age_gender = self.sortGender(X_age)
        if (usePrints):
            print('no age, no gender:',len(X_nan_age_nan_gender.index))
            print('no age, yes gender:',len(X_nan_age_gender.index))
            print('yes age, no gender:',len(X_age_nan_gender.index))
            print('yes age, yes gender:',len(X_age_gender.index))
        X={}
        X['nan_age_nan_gender'] = X_nan_age_nan_gender
        X['nan_age_gender'] = X_nan_age_gender
        X['age_nan_gender'] = X_age_nan_gender
        X['age_gender'] = X_age_gender
        return X
        
    def getGenderIndexs(self):
        indexs = list(range(self.validGenderIndex['n_loc'], self.validGenderIndex['gender_len'] + self.validGenderIndex['n_loc']))
        indexs.remove(self.validGenderIndex['valid_index'] + self.validGenderIndex['n_loc']);
        # print('indexs', indexs)
        return indexs
    
    def fit(self, X_train, y_train):
        if (self.usePrints) : print('-----ageGenderModel FIT-----')
        X_train['y_label'] = y_train
        
        # print('X_trainDF:',X_trainDF.head(20))
        Xsorted = self.sortAgeGender(X_train, usePrints=self.usePrints)
        
        self.y_nan_age_nan_gender = Xsorted['nan_age_nan_gender']['y_label']
        self.X_nan_age_nan_gender = Xsorted['nan_age_nan_gender'].drop(['y_label'], 1)
        # print('y_nan_age_nan_gender:\n',self.y_nan_age_nan_gender.head(2))
        # print('X_nan_age_nan_gender:\n',self.X_nan_age_nan_gender.head(2))
        self.predictMethod0.fit(self.X_nan_age_nan_gender,self.y_nan_age_nan_gender)
        
        self.y_nan_age_gender = Xsorted['nan_age_gender']['y_label']
        self.X_nan_age_gender = Xsorted['nan_age_gender'].drop(['y_label'], 1)
        self.predictMethod1.fit(self.X_nan_age_gender,self.y_nan_age_gender)
        
        self.y_age_nan_gender = Xsorted['age_nan_gender']['y_label']
        self.X_age_nan_gender = Xsorted['age_nan_gender'].drop(['y_label'], 1)
        self.predictMethod2.fit(self.X_age_nan_gender,self.y_age_nan_gender)
        
        self.y_age_gender = Xsorted['age_gender']['y_label']
        self.X_age_gender = Xsorted['age_gender'].drop(['y_label'], 1)
        # print('y_age_gender:\n',self.y_age_gender.head(2))
        # print('X_age_gender:\n',self.X_age_gender.head(2))
        self.predictMethod3.fit(self.X_age_gender,self.y_age_gender)
        if (self.usePrints) : print('-----ageGenderModel END FIT-----')
          
    def predict(self,X_test):
        if (self.usePrints) : print('-----ageGenderModel PREDICT-----')
        testLen = len(X_test['age'])   
        X_test.index = range(testLen)
        XtestSorted = self.sortAgeGender(X_test, usePrints=self.usePrints)
                
        self.X_test_nan_age_nan_gender = XtestSorted['nan_age_nan_gender']
        # print('y_test_nan_age_nan_gender:\n',self.y_test_nan_age_nan_gender.head(2))
        # print('X_test_nan_age_nan_gender:\n',self.X_test_nan_age_nan_gender.head(2))
        y_pred_nan_age_nan_gender = self.predictMethod0.predict(self.X_test_nan_age_nan_gender)
        # print('y_pred_nan_age_nan_gender:\n',y_pred_nan_age_nan_gender[0:2])
        
        self.X_test_nan_age_gender = XtestSorted['nan_age_gender']
        y_pred_nan_age_gender = self.predictMethod1.predict(self.X_test_nan_age_gender)
        
        self.X_test_age_nan_gender = XtestSorted['age_nan_gender']
        y_pred_age_nan_gender = self.predictMethod2.predict(self.X_test_age_nan_gender)
        
        self.X_test_age_gender = XtestSorted['age_gender']
        # print('y_test_age_gender:\n',self.y_test_age_gender.head(2))
        # print('X_test_age_gender:\n',self.X_test_age_gender.head(2))
        y_pred_age_gender = self.predictMethod3.predict(self.X_test_age_gender)
        # print('y_pred_age_gender:\n',y_pred_age_gender[0:2])
        
        y_pred=list(range(testLen))
        for i in range(len(y_pred_nan_age_nan_gender)):
            y_pred[self.X_test_nan_age_nan_gender.index[i]] = y_pred_nan_age_nan_gender[i]
        
        for i in range(len(y_pred_nan_age_gender)):
            y_pred[self.X_test_nan_age_gender.index[i]] = y_pred_nan_age_gender[i] 
        
        for i in range(len(y_pred_age_nan_gender)):
            y_pred[self.X_test_age_nan_gender.index[i]] = y_pred_age_nan_gender[i]
        
        for i in range(len(y_pred_age_gender)):
            y_pred[self.X_test_age_gender.index[i]] = y_pred_age_gender[i]
        if (self.usePrints) : print('-----ageGenderModel END PREDICT-----')
        return y_pred
                
            
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
