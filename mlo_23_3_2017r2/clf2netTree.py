# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:17:00 2017

@author: AlonA
"""
import  clf2net
import random
from sklearn.cross_validation import train_test_split

def clf2netTree(clfList,clfNueral, X_byteDF, y):
    X_train, X_test, y_train, y_test = train_test_split(X_byteDF, y, random_state=1)
    myClf1 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=1)
    myClf2 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=2)
    myClf3 = clf2net.clf2net((myClf1,myClf2), clfNueral, useXtrain = True, random_state=3)
    
    
    myClf4 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=4)
    myClf5 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=5)
    myClf6 = clf2net.clf2net((myClf4,myClf5), clfNueral, useXtrain = True, random_state=6)
    
    
    myClf7 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=7)
    myClf8 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=8)
    myClf9 = clf2net.clf2net((myClf7,myClf8), clfNueral, useXtrain = True, random_state=9)
    
    
    myClf10 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=10)
    myClf11 = clf2net.clf2net((random.choice(clfList),random.choice(clfList)), 
                             clfNueral, splitTrain = True, useXtrain = True, random_state=11)
    myClf12 = clf2net.clf2net((myClf10,myClf11), clfNueral, useXtrain = True, random_state=12)
    
    
    myClf13 = clf2net.clf2net((myClf3,myClf6), clfNueral, useXtrain = True, random_state=13)
    myClf14 = clf2net.clf2net((myClf9,myClf12), clfNueral, useXtrain = True, random_state=14)
    myClf15 = clf2net.clf2net((myClf13,myClf14), clfNueral, useXtrain = True, random_state=15)
    
    myClf15.fit(X_train,y_train)
    myClf15.predict(X_test)
    myClf15.accuracy_score(y_test)
    
    

    