# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 21:27:45 2016

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
from sklearn.neural_network import MLPClassifier

pathTrain = '../input/train_users.csv'
pathTest = '../input/test_users.csv'

users,yRes = orig.runAll(pathTrain,pathTest,'country_destination')
trainFinal =  users[0:len(yRes)]
testFinal  =  users[len(yRes):]
trainFinalfinel = orig.keepColumns(trainFinal,['affiliate_channel', 'first_affiliate_tracked', 'signup_app', 'signup_method', 'signup_flow', 'validAge'])
testFinalfinel = orig.keepColumns(testFinal,['affiliate_channel', 'first_affiliate_tracked', 'signup_app', 'signup_method', 'signup_flow', 'validAge'])

mainPredictMethod = MLPClassifier(solver='lbfgs')
startRun = time.clock()
mainPredictMethod.fit(trainFinalfinel,yRes['country_destination'])
print('fit time:',time.clock()-startRun)
startRun = time.clock()
prediction = mainPredictMethod.predict(testFinalfinel)
print('prediction time:',time.clock()-startRun)

idCol = testFinal['id']
idCol = idCol.reset_index(drop=True)
d = {'id' : idCol, 'country' : pd.Series(prediction)}
df = pd.DataFrame(d,columns=['id', 'country'])
df.to_csv('../submission/geneticFetureTree.csv',index=False)

