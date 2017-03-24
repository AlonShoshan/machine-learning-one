# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:28:15 2017

@author: AlonA
"""

import clf2net
from sklearn.cross_validation import train_test_split
import basicLib.loadAndTest as orig
import basicLib.featureUpdate as featUp

def runAll(path, predictMethodList, predictMethod):
    users = orig.loadAndUpdateFeatures(path)
    featureList = orig.featureList()
    featureList.addByRegex(["action_", "num_of_devices","total_time"],users)
    category = 'country_destination'
    X_byteDF, y = orig.getXbyte(users,featureList.get(),category)
    return runClf2net(predictMethodList,predictMethod, X_byteDF, y)

def runClf2net(predictMethodList,predictMethod, X_byteDF, y, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X_byteDF, y, random_state=1)
    clfList = []
    flags = [{'splitTrain':False, 'useXtrain':False}, {'splitTrain':False, 'useXtrain':True},
             {'splitTrain':True, 'useXtrain':False}, {'splitTrain':True, 'useXtrain':True}]
    for i in flags:
        print('-----------The clf is: splitTrain-', str(i['splitTrain']), ', useXtrain-', str(i['useXtrain']), '-----------')
        myClf = clf2net.clf2net(predictMethodList, predictMethod, 
                                splitTrain = i['splitTrain'], useXtrain = i['useXtrain'])
        myClf.fit(X_train,y_train)
        myClf.predict(X_test)
        myClf.accuracy_score(y_test)
        clfList.append(myClf)
    return clfList