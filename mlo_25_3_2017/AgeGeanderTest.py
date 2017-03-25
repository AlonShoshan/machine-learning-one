# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:03:11 2017

@author: Alon
"""
import time
import basicLib.loadAndTest as orig
import basicLib.ageGenderModel as agm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import numpy as np
import pandas as pd

predictMethod0 = LogisticRegression()
predictMethod1 = LogisticRegression()
predictMethod2 = LogisticRegression()
predictMethod3 = LogisticRegression()

users, yRes = orig.loadAndUpdateFeatures('../input/train_users.csv')
featureListClass = orig.featureList()
print(users.head())
print(yRes.head())
category = 'country_destination'

mainPredictMethod = agm.ageGenderModel(predictMethod0,predictMethod1,predictMethod2,predictMethod3)
users = pd.concat([users,yRes],axis=1)
startRun = time.clock()
prediction, fit = orig.fitPredictAndTest(users,list(featureListClass.get()),category,mainPredictMethod)
print('\n\n\n\n\nnum of featurs',len(list(featureListClass.get())),'run time:',time.clock()-startRun,'fit:',fit)


idCol = finalTest['id']
idCol = idCol.reset_index(drop=True)
d = {'id' : idCol, 'country' : pd.Series(y_pred)}
df = pd.DataFrame(d,columns=['id', 'country'])
df.to_csv('../submission/submission_ageGenderTreeLogistic.csv',index=False)
