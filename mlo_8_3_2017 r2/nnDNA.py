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
import time

def getRandTrueFalse():
    if random.randint(0,1) > 0.5:
        return True
    else:
        return False
    

class DNA:
    def __init__(self, args, genes=None, mutationFactor=0.01):
        self.args = args
        featureListClass = orig.featureList()        
        featureListClass.addByRegex(['action_','num_of_devices'],self.args['users'])
        self.featureList = featureListClass.get()
        self.category = 'country_destination'
        self.mutationFactor = mutationFactor
        self.fitScoreNorm = 0
        self.fitScore = 0
        self.fit = 0
        self.maxNumOfNurons = 100
        self.minAlpaP2 = -6
        self.maxAlpaP2 = 4
        if genes is None :
            self.genes = self.getRandomGenes()
        else :
            self.genes = genes
        self.predictMethod = MLPClassifier(solver='lbfgs', alpha=self.genes[0], hidden_layer_sizes=tuple(self.genes[1:]))
               
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
        return DNA(self.args, genes=newGenes,mutationFactor=self.mutationFactor)
    
    def mutate(self):
        newGenes = []
        newGenes.append(self.mutateAlpa(self.genes[0]))
        newGenes.extend(self.mutateHiddenLyers(self.genes[1:]))
        self.genes = newGenes
                    

    def fitness(self):
        startRun = time.clock()
        self.prediction, self.fit = orig.fitPredictAndTest(self.args['users'],self.featureList,self.category,self.predictMethod,random_state=1)
        print('genes:',self.genes,'run time:',time.clock()-startRun,'fit:',self.fit)
        return self.fit
        
    def calcFitScore(self, fit):
        return exp(5*fit);
        
    def getRandomGenes(self):
        genes = []
        genes.append(self.getRandomAlpa())
        genes.extend(self.getRandomHiddenLayers())
        print('genes:', genes)
        return genes
        
    def getRandomAlpa(self):
        alpaP2 = random.randint(self.minAlpaP2,self.maxAlpaP2)
        return 10 ** (alpaP2) 
        
        
    def getRandomHiddenLayers(self):
        hiddenLayers = []
        hiddenLayers.append(random.randint(1,self.maxNumOfNurons))
        for i in range(2):
            if getRandTrueFalse() == False:
                break
            hiddenLayers.append(random.randint(1,self.maxNumOfNurons))
        return hiddenLayers
        
    def mutateAlpa(self,alpa):
        if random.randint(0, 100) < self.mutationFactor*100 :
            alpa = self.getRandomAlpa()
        return alpa
        
    def mutateHiddenLyers(self,hiddenLayers):
        for i in range(len(hiddenLayers)):
            if random.randint(0, 100) < self.mutationFactor*100 :
                mutation = random.randint(-5,5)
                hiddenLayers[i] = hiddenLayers[i] + mutation
                if hiddenLayers[i] < 1:
                    hiddenLayers[i] = 1
            if ((len(hiddenLayers) < self.maxNumOfNurons) and (random.randint(0, 100) < self.mutationFactor*100)):
                hiddenLayers.append(random.randint(0,self.maxNumOfNurons))
        return hiddenLayers
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        