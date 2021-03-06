# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:44:30 2016

@author: Alon
"""
import random
from math import exp
#import DNA as dna
import FeturesDNA as dna

class population:
    def __init__(self, populationSize=20):
        self.populationSize = populationSize
        self.DNAobjects = []
        for i in range(self.populationSize):
            self.DNAobjects.append(dna.DNA())


    def evaluate(self):
        self.fitList = []
        self.maxFit = 0
        self.maxFitScore = 0
        fitScoreSum = 0
        for i in range(self.populationSize): # geting fit results
            fit = self.DNAobjects[i].fitness()
            self.fitList.append(fit)
            if fit > self.maxFit :
                self.maxFit = fit
        for i in range(self.populationSize): # geting fitScore results
            self.DNAobjects[i].fitScore = self.calcFitScore(self.DNAobjects[i].fit, self.maxFit)
            fitScoreSum += self.DNAobjects[i].fitScore
        for i in range(self.populationSize): # geting normelaizing fitScore results
            self.DNAobjects[i].fitScoreNorm = self.DNAobjects[i].fitScore * 100 / fitScoreSum

    def selection(self):
        newDNAobjects = []
        for i in range(self.populationSize):
            perentA = self.selectDNAobject()
            perentB = self.selectDNAobject(dontSelectPerent=i)
            child = perentA.crossover(perentB)
            child.mutate()
            newDNAobjects.append(child)
        self.DNAobjects = newDNAobjects

    def calcFitScore(self, fit, maxFit):
        power = fit-maxFit
        return exp(20*power)

    def selectDNAobject(self,dontSelectPerent=-1):
        if dontSelectPerent == -1 :
            dontSelectPerent = self.populationSize
        while True:
            perent = random.randint(0, self.populationSize - 1)
            if dontSelectPerent == perent :
                continue
            noSuccses = random.randint(0, 1000)
            if noSuccses < self.DNAobjects[perent].fitScoreNorm*10 :
                return self.DNAobjects[perent]

    def getBestGenes(self):
        for i in range(self.populationSize):
            if self.DNAobjects[i].fitScore == 1 :
                return self.DNAobjects[i].genes

    def getSortedEvaluations(self):
        copyList = self.fitList.copy()
        copyList.sort()
        return copyList
        
    def getSortedPercentage(self):
        fitScoreNormList = []
        for i in range(self.populationSize):
            fitScoreNormList.append(self.DNAobjects[i].fitScoreNorm)
        fitScoreNormList.sort()
        return fitScoreNormList
