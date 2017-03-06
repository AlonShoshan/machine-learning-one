# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:31:08 2017

@author: Alon
"""

from math import exp
import random
from sklearn.neural_network import MLPClassifier
import basicLib.loadAndTest as orig
import numpy as np

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
        self.maxNumOfNurons = 20 # TODO 
        if genes is None :
            self.genes = self.getRandomGenes()
        else :
            self.genes = genes
        self.predictMethod = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=tuple(self.genes))
               
    def crossover(self, foreignDNAobject):
        genesRange = np.maximum(len(foreignDNAobject.genes),len(self.genes))
        newGenes = []
        for i in range(genesRange):
            if  getRandTrueFalse():
                if len(self.genes)-1 < i:
                    break
                newGenes.append(self.genes[i])
            else:
                if len(foreignDNAobject.genes)-1 < i:
                    break
                newGenes.append(foreignDNAobject.genes[i])
        return DNA(newGenes,mutationFactor=self.mutationFactor)
    
    def mutate(self):
        newGenes = list(self.genes)
        for i in range(len(newGenes)):
            if random.randint(0, 100) < self.mutationFactor*100 :
                mutation = random.randint(-5,5)
                newGenes[i] = newGenes[i] + mutation
                if newGenes[i] < 1:
                    newGenes[i] = 1
            if ((len(newGenes) < self.maxNumOfNurons) and (random.randint(0, 100) < self.mutationFactor*100)):
                newGenes.append(random.randint(0,self.maxNumOfNurons))
        self.genes = newGenes
                    

    def fitness(self):
        users = orig.loadAndUpdateFeatures('../input/users_2014_actions_combined_device.csv')
        self.prediction, self.fit = orig.fitPredictAndTest(users,self.featureList,self.category,self.predictMethod,random_state=1)
        return self.fit
        
    def calcFitScore(self, fit):
        return exp(5*fit);
        
    def getRandomGenes(self):
        genes = []        
        genes.append(random.randint(1,self.maxNumOfNurons))
        for i in range(2):
            if getRandTrueFalse() == False:
                break
            genes.append(random.randint(1,self.maxNumOfNurons))
        return genes
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        