# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 16:49:10 2016

@author: Alon
"""
import geneticLib.population as pop
import time

class Evolution:
    def __init__(self, dnaArgs, usePrints=False, populationSize=20,numOfEvolutionSteps=5,mutationFactor=0.01,hotStart=None,bestMovesOn=0):
        self.usePrints = usePrints
        self.populationSize = populationSize
        self.evolutionStep = 0
        self.numOfEvolutionSteps = numOfEvolutionSteps
        self.population = pop.population(dnaArgs, populationSize=self.populationSize,hotStart=hotStart,bestMovesOn=bestMovesOn)
        self.mutationFactor = mutationFactor
        self.Best = {'Genes' : None, 'Fit' : 0}

    def run(self):
        self.startRun = time.clock() 
        self.printStart()
        for evStep in range(self.numOfEvolutionSteps):
            startStep = time.clock()
            self.evolutionStep = evStep
            if self.usePrints : print('-------------STARTING EVOLUTION',evStep,' STEP-------------')
            self.population.evaluate()
            if self.usePrints :
                print('Sorted evaluations:',self.population.getSortedEvaluations())
                print('Sorted percentage:',self.population.getSortedPercentage())
                print('Best genes (fit:',self.population.getSortedEvaluations()[self.populationSize-1],') in Evolution Step:',self.population.getBestGenes())
            self.rememberBestGenes()
            if evStep != self.numOfEvolutionSteps -1 :
                self.population.selection()
            if self.usePrints :
                print('Best ovrall Genes:',self.Best)
                print('Evolution Step douration:', time.clock() - startStep)
        self.printEnd()
            
    def rememberBestGenes(self):
        if self.Best['Fit'] < self.population.maxFit :
            self.Best['Fit'] = self.population.maxFit
            self.Best['Genes'] = self.population.getBestGenes().copy()
            
    def printStart(self):
        if self.usePrints :
            print('------------------------------START EVOLUTION--------------------------------------')
            print('Population Size', self.populationSize)
            print('Mutation Factor Size', self.mutationFactor)
            print('-----------------------------------------------------------------------------------')
    
    def printEnd(self):
        if self.usePrints :
            print('----------------------------END OF EVOLUTION--------------------------------------')
            print('Evolution douration:', time.clock() - self.startRun)
            print('Best Result in Last Evolution Step:', self.population.getSortedEvaluations()[self.populationSize-1], ', Worst Result in Last Evolution Step:', self.population.getSortedEvaluations()[0])
            print('Best ovrall Genes:',self.Best)            
            print('-----------------------------------------------------------------------------------')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
