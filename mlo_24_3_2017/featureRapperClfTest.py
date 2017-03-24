# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:33:24 2017

@author: Alon
"""
import featureRapperClf as frc
import basicLib.loadAndTest as orig
from sklearn import tree
import time

predictMethod = tree.DecisionTreeClassifier()
users = orig.loadAndUpdateFeatures('../input/users_2014_sessions_norm.csv')

featureListClass = orig.featureList(usersCol=users.columns)        
featureListAll = featureListClass.get()

category = 'country_destination'
featureList = ['action_set_password_##_submit_##_set_password','action_authenticate_##_submit_##_login']

featureRapper = frc.featureRapperClf(predictMethod, featureList)

startRun = time.clock()
prediction, fit = orig.fitPredictAndTest(users,featureListAll,category,featureRapper,random_state=1)
print('run time:',time.clock()-startRun,'fit:',fit)