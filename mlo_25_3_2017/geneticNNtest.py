# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:51:35 2017

@author: Alon
"""

import nnDNA as dna
import geneticLib.Evolution as Evo
import basicLib.loadAndTest as orig
import pandas as pd

users , yRes = orig.loadAndUpdateFeatures('../input/train_users.csv')
users = pd.concat([users,yRes],axis=1)
args = {'users' : users}
dnaArg = {'dna' : dna, 'args' : args}
Ev = Evo.Evolution(dnaArg, usePrints=True, populationSize=20, numOfEvolutionSteps=60,mutationFactor=0.1,bestMovesOn=2)
Ev.run()
