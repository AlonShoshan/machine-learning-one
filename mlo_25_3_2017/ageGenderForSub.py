# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:50:44 2017

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

pathTrain = '../input/users_train_final.csv'
pathTest = '../input/users_test_final.csv'
pathyRes = '../input/yRes_final.csv'
finalTrain = pd.read_csv(pathTrain)
finalTest = pd.read_csv(pathTest)
yRes = pd.read_csv(pathyRes)
finalTrainUse = orig.removeColumns(finalTrain,("action_", "num_of_devices","total_time","timeAct_","dev_","id"))
finalTestUse = orig.removeColumns(finalTest,("action_", "num_of_devices","total_time","timeAct_","dev_","id"))

predList = []
for j in range(3):
    predList.append(LogisticRegression())
predList.append(MLPClassifier(solver='lbfgs'))
mainPredictMethod = agm.ageGenderModel(predList[0],predList[1],predList[2],predList[3])
startRun = time.clock()
mainPredictMethod.fit(finalTrainUse,yRes['country_destination'])
print('fit time:',time.clock()-startRun)
startRun = time.clock()
prediction = mainPredictMethod.predict(finalTestUse)
print('prediction time:',time.clock()-startRun)

idCol = finalTest['id']
idCol = idCol.reset_index(drop=True)
d = {'id' : idCol, 'country' : pd.Series(prediction)}
df = pd.DataFrame(d,columns=['id', 'country'])
df.to_csv('../submission/submission_ageGenderTree_LLLN.csv',index=False)

