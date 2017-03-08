# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 12:56:34 2017

@author: Alon
"""
import DNAexample as dna
import Evolution as Evo

Ev = Evo.Evolution(dna,usePrints=True,populationSize=20,numOfEvolutionSteps=100)
Ev.run()