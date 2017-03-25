# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:44:42 2017

@author: Alon
"""

import time
import basicLib.loadAndTest as orig
import basicLib.ageGenderModel as agm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import numpy as np
import pandas as pd

users, yRes = orig.loadAndUpdateFeatures('../input/train_users.csv')
featureListClass = orig.featureList()
category = 'country_destination'

users = pd.concat([users,yRes],axis=1)

print('0 nn\n')
for i in range(1):
    predList = []
    for j in range(4):
        predList.append(LogisticRegression())
    mainPredictMethod = agm.ageGenderModel(predList[0],predList[1],predList[2],predList[3])
    startRun = time.clock()
    prediction, fit = orig.fitPredictAndTest(users,list(featureListClass.get()),category,mainPredictMethod)
    print('------------------0nn--------------------------------------------\nnum of featurs',len(list(featureListClass.get())),'run time:',time.clock()-startRun,'fit:',fit,'\n-----------------------------i:',i,'j:',j,'--------------------------------------')



print('1 nn\n')
for i in range(4):
    predList = []
    for j in range(4):
        if i == j:
            predList.append(MLPClassifier(solver='lbfgs'))
        else:
            predList.append(LogisticRegression())
    mainPredictMethod = agm.ageGenderModel(predList[0],predList[1],predList[2],predList[3])
    startRun = time.clock()
    prediction, fit = orig.fitPredictAndTest(users,list(featureListClass.get()),category,mainPredictMethod)
    print('------------------1nn--------------------------------------------\nnum of featurs',len(list(featureListClass.get())),'run time:',time.clock()-startRun,'fit:',fit,'\n-----------------------------i:',i,'j:',j,'--------------------------------------')


print('3 nn\n')
for i in range(4):
    predList = []
    for j in range(4):
        if i == j:
            predList.append(LogisticRegression())
        else:
            predList.append(MLPClassifier(solver='lbfgs'))
    mainPredictMethod = agm.ageGenderModel(predList[0],predList[1],predList[2],predList[3])
    startRun = time.clock()
    prediction, fit = orig.fitPredictAndTest(users,list(featureListClass.get()),category,mainPredictMethod)
    print('--------------------3nn------------------------------------------\nnum of featurs',len(list(featureListClass.get())),'run time:',time.clock()-startRun,'fit:',fit,'\n-----------------------------i:',i,'j:',j,'--------------------------------------')

print('4 nn\n')
for i in range(1):
    predList = []
    for j in range(4):
        predList.append(MLPClassifier(solver='lbfgs'))
    mainPredictMethod = agm.ageGenderModel(predList[0],predList[1],predList[2],predList[3])
    startRun = time.clock()
    prediction, fit = orig.fitPredictAndTest(users,list(featureListClass.get()),category,mainPredictMethod)
    print('------------------4nn--------------------------------------------\nnum of featurs',len(list(featureListClass.get())),'run time:',time.clock()-startRun,'fit:',fit,'\n-----------------------------i:',i,'j:',j,'--------------------------------------')





























































