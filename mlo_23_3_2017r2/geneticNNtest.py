# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:51:35 2017

@author: Alon
"""

import nnDNA as dna
import geneticLib.Evolution as Evo
import basicLib.loadAndTest as orig

users = orig.loadAndUpdateFeatures('../input/users_2014_sessions_norm.csv')
args = {'users' : users}
dnaArg = {'dna' : dna, 'args' : args}
Ev = Evo.Evolution(dnaArg, usePrints=True, populationSize=6, numOfEvolutionSteps=10,mutationFactor=0.1)
Ev.run()
