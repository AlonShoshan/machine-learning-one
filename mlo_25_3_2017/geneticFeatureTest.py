# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:41:01 2017

@author: Alon
"""

import featureDNA as dna
import geneticLib.Evolution as Evo
import basicLib.loadAndTest as orig
from sklearn import tree
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

predictMethod = MLPClassifier(solver='lbfgs')
#predictMethod = LogisticRegression()
users , yRes = orig.loadAndUpdateFeatures('../input/train_users.csv')
users = pd.concat([users,yRes],axis=1)
args = {'users' : users, 'predictMethod' : predictMethod}
dnaArg = {'dna' : dna, 'args' : args}
features1 = ['gender']
features2 = ['affiliate_provider','first_browser','first_device_type','gender','language','signup_app','signup_method','signup_flow']
featureListClass = orig.featureList()
features3 = featureListClass.get()
genesList = [features1,features2,features3]
hotStart = {'numOfGenes' : len(genesList), 'genesList' : genesList} # genes is a list of list (list of list of feturs)
Ev = Evo.Evolution(dnaArg,usePrints=True,populationSize=20,numOfEvolutionSteps=60,hotStart=hotStart,bestMovesOn=2)
Ev.run()
