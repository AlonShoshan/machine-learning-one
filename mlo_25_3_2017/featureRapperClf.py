# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:08:20 2017

@author: Alon
"""
##############################################################
# featureRapperClf is a repper Class for any classefier so   #
# we can esaly sort the features that this clasifer will use #
# users provided predictMethod and wanted featureList        #
# from here will work exaktly like normal classifer          #
##############################################################
class featureRapperClf:
    def __init__(self, predictMethod, featureList):
        self.predictMethod = predictMethod
        self.featureList = featureList
           
    def fit(self, X_train, y_train):
        X_train_sortedFeatuers = self.sortFeatures(X_train.copy(deep=True))
        self.predictMethod.fit(X_train_sortedFeatuers, y_train)
        
    def predict(self,X_test):
        X_test_sortedFeatuers = self.sortFeatures(X_test.copy(deep=True))
        return self.predictMethod.predict(X_test_sortedFeatuers)
        
    def predict_proba(self,X_test):
        return self.predictMethod.predict_proba(X_test)
        
    def sortFeatures(self, X_train) :
        X_train_sortedFeatuers = X_train[self.featureList]
        return X_train_sortedFeatuers
