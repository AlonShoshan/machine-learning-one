# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:41:01 2017

@author: Alon
"""

import featureDNA as dna
import geneticLib.Evolution as Evo

Ev = Evo.Evolution(dna,usePrints=True,populationSize=15,numOfEvolutionSteps=10)
Ev.run()