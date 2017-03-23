# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:38:11 2016

@author: Alon
"""

import numpy as np
import pandas as pd
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
from sklearn.neural_network import MLPClassifier
import basicLib.loadAndTest as orig
import basicLib.ageGenderModel as ageGender
import random

# runs N clsification models and returning there predictions and accuracy
# users: data
# model_list: list (sizes N) of dictoinaris - {'PredictMethod' : model_num, 'featureList' : modelFeatureList, 'category' : category} 
# wile PredictMethod: {0 : logrg, 1 : decisionTree, 2 : neural, 3 : ageGenderLogreg, 4 : ageGenderTree, 5 : ageGenderRand} 
def runModuls (users, model_list, random_state=1, usePrints=False):
    if (usePrints):
        print('-----------------RUN MODELS-------------------')
        print('number of models:', len(model_list))
        print('-----------------START------------------------')
    y_pred_list = []
    accuracy_score_list = []
    model_num = 0
    for model in model_list:
        copy_users = users.copy()
        modelName = getPredictMethodName(model['PredictMethod'])
        if ((modelName in ['ageGenderLogreg','ageGenderTree','ageGenderRand'])) :
            if (False == ('gender' in model['featureList'])) :
                model['featureList'].append('gender')
            if (False == ('validAge' in model['featureList'])) :
                model['featureList'].append('validAge')
            if (False == ('age' in model['featureList'])) :
                model['featureList'].append('age')
        if (usePrints):
            print('-----------------MODEL',model_num,':',modelName,'-------------------')
        predictMethod = getPredictMethod(model['PredictMethod'],random_state=random_state,usePrints=usePrints)
        y_pred, accuracy_score = orig.fitPredictAndTest(copy_users,model['featureList'],
                                                  model['category'],predictMethod,random_state=random_state,usePrints=usePrints)
        y_pred_list.append(y_pred)
        accuracy_score_list.append(accuracy_score)
        model_num += 1
    return y_pred_list, accuracy_score_list  
    
# private
def getPredictMethod(modelNum,random_state=1,usePrints=False):
    def logrg():
        return LogisticRegression()
    def decisionTree():
        return tree.DecisionTreeClassifier()
    def neural():
        return MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=random_state)
    def ageGenderLogreg():
        return ageGender.ageGenderModel(getPredictMethod(0),getPredictMethod(0),getPredictMethod(0),getPredictMethod(0))
    def ageGenderTree():
        return ageGender.ageGenderModel(getPredictMethod(1),getPredictMethod(1),getPredictMethod(1),getPredictMethod(1))
    def ageGenderRand():
        modelindex = list(range(4))
        for i in range(4):
            modelindex[i] = random.randint(0, 2)
        if usePrints : print(getPredictMethodName(modelindex[0]),getPredictMethodName(modelindex[1]),getPredictMethodName(modelindex[2]),getPredictMethodName(modelindex[3]))
        return ageGender.ageGenderModel(getPredictMethod(modelindex[0]),getPredictMethod(modelindex[1]),getPredictMethod(modelindex[2]),getPredictMethod(modelindex[3]))
    options = {0 : logrg, 1 : decisionTree, 2 : neural, 3 : ageGenderLogreg, 4 : ageGenderTree, 5 : ageGenderRand}
    return options[modelNum]()

# private
def getPredictMethodName(modelNum,random_state=1):
    def logrg():
        return 'logreg'
    def decisionTree():
        return 'decisionTree'
    def neural():
        return 'neural ='
    def ageGenderLogreg():
        return 'ageGenderLogreg'
    def ageGenderTree():
        return 'ageGenderTree'
    def ageGenderRand():
        return 'ageGenderRand'
    options = {0 : logrg, 1 : decisionTree, 2 : neural, 3 : ageGenderLogreg, 4 : ageGenderTree, 5 : ageGenderRand}
    return options[modelNum]()
    
def randomModel(featureList, numOfModeld=6, numOfModelsToCreate=10, category='country_destination',ModelMethod=-1):
    model_list = []
    for model in range(numOfModelsToCreate):
        modelFeatureList = orig.featureList.randomList(featureList)
        if (ModelMethod == -1) :
            model_num = random.randint(0, numOfModeld - 1)
        else :
            model_num = ModelMethod
        model = {'PredictMethod' : model_num, 'featureList' : modelFeatureList, 'category' : category}
        model_list.append(model)
    return model_list