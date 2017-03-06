# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:51:35 2017

@author: Alon
"""

import nnDNA as dna
import geneticLib.Evolution as Evo

Ev = Evo.Evolution(dna,usePrints=True,populationSize=8,numOfEvolutionSteps=10,mutationFactor=0.1)
Ev.run()