# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:27:02 2016

@author: Alon
"""
from math import exp
import random
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import basicLib.originalFunctions3 as orig

def getRandTrueFalse():
    if random.randint(0,1) > 0.5:
        return True
    else:
        return False
    

class DNA:
    def __init__(self, genes=None, mutationFactor=0.01):
        self.featureList = [ 'affiliate_channel','affiliate_provider','first_affiliate_tracked','first_browser','first_device_type','language',
                'signup_app','signup_method','signup_flow','date_account_created_month','date_first_active_month','date_account_created_dayofweek',
                'date_first_active_dayofweek','age','validAge','gender' ]
        self.category = 'country_destination'
        self.mutationFactor = mutationFactor
        self.fitScoreNorm = 0
        self.fitScore = 0
        self.fit = 0
        if genes is None :
            self.genes = []
            for i in range(len(self.featureList)):
                if getRandTrueFalse():
                    self.genes.append(self.featureList[i])
        else :
            self.genes = genes
        # self.predictMethod = LogisticRegression()
        self.predictMethod = tree.DecisionTreeClassifier()
        

    def crossover(self, foreignDNAobject):
        newGenes = []
        for i in range(len(self.featureList)):
            if  getRandTrueFalse():
                if self.featureList[i] in self.genes:
                    newGenes.append(self.featureList[i])
            else:
                if self.featureList[i] in foreignDNAobject.genes:
                    newGenes.append(self.featureList[i])
        return DNA(newGenes,mutationFactor=self.mutationFactor)

    def mutate(self):
        newGenes = self.genes
        for i in range(len(self.featureList)):
            if random.randint(0, 100) < self.mutationFactor*100 :
                if self.featureList[i] in self.genes:
                    newGenes.remove(self.featureList[i])
                else:
                    newGenes.append(self.featureList[i])
        self.genes = newGenes
                    

    def fitness(self):
        users = orig.loadAndUpdateAirBnbUsers('input/train_users.csv',13,95,addValidAgeBit=True)
        self.prediction, self.fit = orig.fitPredictAndTest(users,self.genes,self.category,self.predictMethod,random_state=1)
        return self.fit
        
    def calcFitScore(self, fit):
        return exp(5*fit);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        