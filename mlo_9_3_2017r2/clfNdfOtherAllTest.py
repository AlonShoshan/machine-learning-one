# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:37:46 2017

@author: Alon
"""

import basicLib.loadAndTest as orig
from sklearn.neural_network import MLPClassifier
import clfNdfOtherAll as clfAll
from sklearn import tree


users = orig.loadAndUpdateFeatures('../input/users_2014_actions_combined_device.csv')
featureListClass = orig.featureList()        
featureListClass.addByRegex(['action_','num_of_devices'],users)
featureList = featureListClass.get()
category = 'country_destination'
        
predictMethod1 = MLPClassifier(solver='lbfgs', alpha=1e-5)
predictMethod2 = MLPClassifier(solver='lbfgs', alpha=1e-5)
predictMethod3 = tree.DecisionTreeClassifier()
clfNdfOtherAllClass = clfAll.clfNdfOtherAll(predictMethod1,predictMethod3)


prediction, fit, y_test = orig.fitPredictAndTest(users,featureList,category,clfNdfOtherAllClass,random_state=1)
print(fit)
print(clfNdfOtherAllClass.accuracy_score(y_test))
