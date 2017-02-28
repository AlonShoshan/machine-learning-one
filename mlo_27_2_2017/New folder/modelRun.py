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
import original_functions.originalFunctions3 as orig
import original_functions.ageGenderModel as ageGender
import random

def runModuls (users, model_list, random_state, usePrints=False):
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
            if (False == ('gender' in model['festureList'])) :
                model['festureList'].append('gender')
            if (False == ('validAge' in model['festureList'])) :
                model['festureList'].append('validAge')
            if (False == ('age' in model['festureList'])) :
                model['festureList'].append('age')
        if (usePrints):
            print('-----------------MODEL',model_num,':',modelName,'-------------------')
        predictMethod = getPredictMethod(model['PredictMethod'],random_state=random_state,usePrints=usePrints)
        y_pred, accuracy_score = orig.fitPredictAndTest(copy_users,model['festureList'],
                                                  model['category'],predictMethod,random_state=random_state)
        y_pred_list.append(y_pred)
        accuracy_score_list.append(accuracy_score)
        model_num += 1
    return y_pred_list, accuracy_score_list  
    


def getPredictMethod(modelNum,random_state=1,usePrints=False):
    def logrg():
        return LogisticRegression()
    def decisionTree():
        return tree.DecisionTreeClassifier()
    def nural():
        return MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=random_state)
    def ageGenderLogreg():
        return ageGender.ageGenderModel(getPredictMethod(0),getPredictMethod(0),getPredictMethod(0),getPredictMethod(0))
    def ageGenderTree():
        return ageGender.ageGenderModel(getPredictMethod(1),getPredictMethod(1),getPredictMethod(1),getPredictMethod(1))
    def ageGenderRand():
        import original_functions.ageGenderModel as ageGender
        modelindex = list(range(4))
        for i in range(4):
            modelindex[i] = random.randint(0, 2)
        if usePrints : print(getPredictMethodName(modelindex[0]),getPredictMethodName(modelindex[1]),getPredictMethodName(modelindex[2]),getPredictMethodName(modelindex[3]))
        return ageGender.ageGenderModel(getPredictMethod(modelindex[0]),getPredictMethod(modelindex[1]),getPredictMethod(modelindex[2]),getPredictMethod(modelindex[3]))
    options = {0 : logrg, 1 : decisionTree, 2 : nural, 3 : ageGenderLogreg, 4 : ageGenderTree, 5 : ageGenderRand}
    return options[modelNum]()

def getPredictMethodName(modelNum,random_state=1):
    def logrg():
        return 'logreg'
    def decisionTree():
        return 'decisionTree'
    def nural():
        return 'nural'
    def ageGenderLogreg():
        return 'ageGenderLogreg'
    def ageGenderTree():
        return 'ageGenderTree'
    def ageGenderRand():
        return 'ageGenderRand'
    options = {0 : logrg, 1 : decisionTree, 2 : nural, 3 : ageGenderLogreg, 4 : ageGenderTree, 5 : ageGenderRand}
    return options[modelNum]()
    
def randomModel(featureList, numOfModeld=6, numOfModelsToCreate=10, category='country_destination',ModelMethod=-1):
    model_list = []
    for model in range(numOfModelsToCreate):
        modelFeatureList = randomList(featureList)
        if (ModelMethod == -1) :
            model_num = random.randint(0, numOfModeld - 1)
        else :
            model_num = ModelMethod
        model = {'PredictMethod' : model_num, 'festureList' : modelFeatureList, 'category' : category}
        model_list.append(model)
    return model_list
    
def randomList(featureList,minFeaturs=4):
    numOfFeature = len(featureList)
    numOfFeautersInNewModel = random.randint(minFeaturs, numOfFeature)
    newFeatureListIndexes = random.sample(range(0, numOfFeature), numOfFeautersInNewModel)
    newFeatureList = [featureList[index] for index in newFeatureListIndexes]
    return newFeatureList