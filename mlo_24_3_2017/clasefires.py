# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 18:56:57 2017

@author: Alon
"""

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")

from sklearn.linear_model import Perceptron
clf = Perceptron()

from sklearn.linear_model import PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()

from sklearn.kernel_ridge import KernelRidge
clf = KernelRidge()

from sklearn import svm
clf = svm.SVC()
clf = svm.LinearSVC() 

from sklearn.neighbors import KNeighborsClassifier
cls = KNeighborsClassifier()

from sklearn.gaussian_process import GaussianProcessRegressor
cls = GaussianProcessRegressor()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

from sklearn.naive_bayes import BernoulliNB
mnb = BernoulliNB()

from sklearn import tree
clf = tree.DecisionTreeClassifier()

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(base_estimator=None) # None - suld be a outer clsiffier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()

from sklearn.ensemble import RandomTreesEmbedding
clf = RandomTreesEmbedding()

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)

## need to continue from ambeded 









