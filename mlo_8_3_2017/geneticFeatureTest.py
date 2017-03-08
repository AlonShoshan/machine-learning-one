# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:41:01 2017

@author: Alon
"""

import featureDNA as dna
import geneticLib.Evolution as Evo
import basicLib.loadAndTest as orig

users = orig.loadAndUpdateFeatures('../input/users_2014_actions_combined_device.csv')
args = {'users' : users}
dnaArg = {'dna' : dna, 'args' : args}
Ev = Evo.Evolution(dnaArg,usePrints=True,populationSize=20,numOfEvolutionSteps=15)
Ev.run()
