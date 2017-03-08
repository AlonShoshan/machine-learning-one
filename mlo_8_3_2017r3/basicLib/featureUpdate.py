# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:03:02 2017

@author: AlonA
"""

import numpy as np
import pandas as pd

#generate the default FeatureStruct
def generateFeatureStruct():
    featureStruct = []
    featureStruct.append(ageFeature())
    featureStruct.append(genderFeature())
    featureStruct.append(first_affiliate_trackedFeature())
    featureStruct.append(countriesFeature())
    featureStruct.append(categoryFeature())
    featureStruct.append(date_account_createdFeature())
    featureStruct.append(date_first_bookingFeature())
    featureStruct.append(date_first_activeFeature())
    return featureStruct

    
##############################################
# Each feature class responssible of the     #
# changes that has to be made to the feature #
# Each class will have an 'update' method    #
# that includes the changes made to the      #
# users dataframe                            #
##############################################
 
class categoryFeature():
    def __init__(self, categorical_features=('affiliate_channel',
                                                   'affiliate_provider',
                                                   'country_destination',
                                                   'NDF_OTHER_destination',
                                                   'NDF_US_OTHER_destination',
                                                   'first_affiliate_tracked',
                                                   'first_browser',
                                                   'first_device_type',
                                                   'gender',
                                                   'language',
                                                   'signup_app',
                                                   'signup_method',
                                                   'signup_flow')            ):
        self.categorical_features = list(categorical_features)
    def update(self,users):
        for categorical_feature in self.categorical_features:
            users[categorical_feature] = users[categorical_feature].astype('category')

class ageFeature():
    def __init__(self, youngTreshold=15, oldTreshold=85, normalizeAge=True ,addValidAgeBit=True):
        self.youngTreshold = youngTreshold
        self.oldTreshold = oldTreshold
        self.normalizeAge = normalizeAge
        self.addValidAgeBit = addValidAgeBit
    def update(self,users):
        users.loc[users.age > self.oldTreshold, 'age'] = np.nan
        users.loc[users.age < self.youngTreshold, 'age'] = np.nan
        users.age.replace(np.nan,'-unknown-', inplace=True)
        if (self.addValidAgeBit):
            users['validAge'] = 0
            users.loc[users.age != '-unknown-', 'validAge'] = 1
            users.loc[users.age == '-unknown-', 'age'] = users[users.age != '-unknown-'].age.mean()  
        if self.normalizeAge : users.age = users.age / (self.oldTreshold + 1)
        users['age'] = users['age'].astype('float64')

class genderFeature():
    def __init__(self,addValidGenderBit=True):
        self.addValidGenderBit = addValidGenderBit
    def update(self,users):
        users.gender.replace(np.nan,'-unknown-', inplace=True)
        if (self.addValidGenderBit):
            users['validGender'] = 0
            users.loc[users.gender != '-unknown-', 'validGender'] = 1
        
class first_affiliate_trackedFeature():
    def __init__(self):
        pass
    def update(self,users):
        users.first_affiliate_tracked.replace(np.nan,'-unknown-', inplace=True)

class countriesFeature():
    def __init__(self):
        pass
    def update(self,users):
        users['NDF_US_OTHER_destination'] = 0
        users.loc[(users.country_destination == 'NDF'), 'NDF_US_OTHER_destination'] = 'NDF'
        users.loc[(users.country_destination == 'US'), 'NDF_US_OTHER_destination'] = 'US'
        users.loc[(users.country_destination != 'NDF') & (users.country_destination != 'US'), 'NDF_US_OTHER_destination'] = 'other'
        users['NDF_OTHER_destination'] = 0
        users.loc[(users.country_destination == 'NDF'), 'NDF_OTHER_destination'] = 'NDF'
        users.loc[(users.country_destination != 'NDF'), 'NDF_OTHER_destination'] = 'other'
        
class date_account_createdFeature():
    def __init__(self):
        pass
    def update(self,users):
        users['date_account_created'] = pd.to_datetime(users['date_account_created'])
    
class date_first_bookingFeature():
    def __init__(self):
        pass
    def update(self,users):
        users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
        
class date_first_activeFeature():
    def __init__(self):
        pass
    def update(self,users):
        users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
        del users['timestamp_first_active']
