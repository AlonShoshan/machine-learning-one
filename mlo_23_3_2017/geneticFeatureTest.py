# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:41:01 2017

@author: Alon
"""

import featureDNA as dna
import geneticLib.Evolution as Evo
import basicLib.loadAndTest as orig
from sklearn import tree

predictMethod = tree.DecisionTreeClassifier()
users = orig.loadAndUpdateFeatures('../input/users_2014_sessions_norm.csv')
args = {'users' : users, 'predictMethod' : predictMethod}
dnaArg = {'dna' : dna, 'args' : args}
features1 = ['action_toggle_archived_thread_##_click_##_toggle_archived_thread']
features2 = ['affiliate_provider','first_affiliate_tracked','first_browser','first_device_type','gender','language','signup_app','signup_method','signup_flow']
genesList = [features1,features2]
hotStart = {'numOfGenes' : len(genesList), 'genesList' : genesList} # genes is a list of list (list of list of feturs)
Ev = Evo.Evolution(dnaArg,usePrints=True,populationSize=20,numOfEvolutionSteps=60,hotStart=hotStart)
Ev.run()
