# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 16:05:19 2017

@author: AlonA
"""

import basicLib.originalFunctions4 as orig
import basicLib.featureUpdate as featUp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import *
from pylab import *
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import re
from sklearn.linear_model import LogisticRegression
import basicLib.modelRun as runms

path = 'input/users_2014_actions_combined_device.csv'
users = orig.loadAndUpdateFeatures(path)
featureList = orig.featureList()
featureList.addByRegex(["action_", "num_of_devices"],users)
predictMethod = LogisticRegression()
category = 'country_destination'
featureList.remove( ['affiliate_channel','affiliate_provider','first_affiliate_tracked',
                           'first_browser','first_device_type','language','signup_app','signup_method',
                           'signup_flow','date_account_created_month','date_first_active_month',
                           'date_account_created_dayofweek','date_first_active_dayofweek','age',
                           'validAge','gender', 'validGender' ])
model_list=[{'PredictMethod' : 0, 'featureList' : featureList.get(), 'category' : category}]
y_pred_list, accuracy_score_list = runms.runModuls(users, model_list, 10, usePrints=True)