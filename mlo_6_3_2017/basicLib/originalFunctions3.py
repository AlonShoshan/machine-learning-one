# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:24:13 2016

@author: Alon
"""

# virsion 1.3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import *
from pylab import *
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# sort wanted Features
def sortFeatures(users,festureList):
    return users[festureList]
# sort wanted Results
def sortResults(users,category):
    return users[category]
# initiate the users
def initiateUsers(users,usePrints=False):
    dateFeaturs(users,Month=True,DayOfWeek=True)

class fitPredictAndTestClass :
    def __init__(self,usePrints=False,crossPredict=True,random_state=1):
        self.usePrints = usePrints
        self.crossPredict = crossPredict
        self.random_state=random_state
        
    def trainTestSplit(self,users,festureList,category,random_state=1):
        self.festureList = festureList
        initiateUsers(users)
        X = sortFeatures(users,festureList)
        X_byteDF = category2binaryFeatures(X, usePrints=self.usePrints)
        y = sortResults(users,category)
        # print('y:',y.head(20))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_byteDF, y, random_state=self.random_state)
        
    def PredictAndTest(self,predictMethod,labels=[],crossPredict=False):
        X_train = self.sortX(self.X_train,self.festureList)
        X_test = self.sortX(self.X_test,self.festureList)
        predictMethod.fit(X_train,self.y_train)
        y_pred = predictMethod.predict(X_test)
        accuracy_score = metrics.accuracy_score(self.y_test, y_pred)
        if (self.usePrints):
            print('accuracy_score:',accuracy_score)
            print('y_pred[0:8] :',y_pred[0:8])
            print('y_test[0:8] :',self.y_test[0:8])
        
        y_pred_log = predictMethod.predict_log_proba(X_test)
        y_pred_proba = predictMethod.predict_proba(X_test)
        return {'y_pred' : y_pred, 'y_pred_log' : y_pred_log, 'y_pred_proba' : y_pred_proba, 'accuracy_score' : accuracy_score}
        
    def sortX(self,X,labels):
        colomns = []
        for label in labels:
            if label in X.columns :
                colomns.append(label)
            index = 0
            while(True):
                if (label + '_' + str(index)) in X.columns :
                    colomns.append(label + '_' + str(index))
                    index += index
                else:
                    break                
        return X[colomns] 
        
        
        
   

# fit predict test acording to predictMethod
def fitPredictAndTest(users,featureList,category,predictMethod,usePrints=False,random_state=1,needPreProssesing=False,crossPredict=False):
    initiateUsers(users)
    X = sortFeatures(users,featureList)
    X_byteDF = category2binaryFeatures(X, usePrints=usePrints)
    y = sortResults(users,category)
    # print('y:',y.head(20))
    X_train, X_test, y_train, y_test = train_test_split(X_byteDF, y, random_state=random_state)
    predictMethod.fit(X_train,y_train)
    y_pred = predictMethod.predict(X_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    if (usePrints):
        print('accuracy_score:',accuracy_score)
        print('y_pred[0:8] :',y_pred[0:8])
        print('y_test[0:8] :',y_test[0:8])
    if (crossPredict) :
        y_pred_log = predictMethod.predict_log_proba(X_test)
        y_pred_proba = predictMethod.predict_proba(X_test)
        return y_pred, accuracy_score, y_pred_log, y_pred_proba 
    return y_pred, accuracy_score

# add date related features
def dateFeaturs(users,Month=False,DayOfWeek=False,DayOfYear=False,WeekofYear=False,Quarter=False,account_created=True,first_active=True):
    if account_created :
        add_date('date_account_created',users,Month=Month,DayOfWeek=DayOfWeek,DayOfYear=DayOfYear,WeekofYear=WeekofYear,Quarter=Quarter)
    if first_active    : 
        add_date('date_first_active',users,Month=Month,DayOfWeek=DayOfWeek,DayOfYear=DayOfYear,WeekofYear=WeekofYear,Quarter=Quarter)

# add date account created features
def add_date(dateFeature,users,Month=False,DayOfWeek=False,DayOfYear=False,WeekofYear=False,Quarter=False):
    month = []
    dayofweek = []
    dayofyear = []
    weekofyear = []
    quarter = []
    for date in getattr(users,dateFeature):
        if Month : month.append(date.month)
        if DayOfWeek : dayofweek.append(date.dayofweek)
        if DayOfYear : dayofyear.append(date.dayofyear)
        if WeekofYear : weekofyear.append(date.weekofyear)
        if Quarter : quarter.append(date.quarter)
    if Month : users[dateFeature + '_month'] = pd.Series(month)
    if DayOfWeek : users[dateFeature + '_dayofweek'] = pd.Series(dayofweek)
    if DayOfYear : users[dateFeature + '_dayofyear'] = pd.Series(dayofyear)
    if WeekofYear : users[dateFeature + '_weekofyear'] = pd.Series(weekofyear)
    if Quarter : users[dateFeature + '_quarter'] = pd.Series(quarter)



# transforme categoryal features to binery features
def category2binaryFeatures(Xlabel, usePrints=False, addAge=False):
    if ((Xlabel.columns == 'age').any()): # if users have age bit 
        userAge = Xlabel[['age']]
        addAge = True
        Xlabel = Xlabel.drop('age', 1)
    
    labels = Xlabel.columns
    myLabels = {} # holds the encode labels with value between 0 and n-1
    n_values = [] # holds the length of each feature in bits ('gender' with 4 option will be 4 bits)
    X = []
    for label in Xlabel:
        labelAttr = getattr(Xlabel, label)
        myLabels[label] = preprocessing.LabelEncoder()
        myLabels[label].fit(labelAttr)
        if size(labelAttr.unique()) > 400:
            print('The feature:', label, 'is too big to handle as binary')
            sys.exit(0)
        n_values.append(size(labelAttr.unique()))
        #transform: turn options into 0 to n-1,  reshape+transpose: align the coulumn so it be easier to add them to the table 
        X.append(myLabels[label].transform(labelAttr).reshape(1,len(labelAttr)).transpose())
    
    temp = np.array([]).reshape(len(X[0]),0)
    for i in range(len(X)):
        temp = concatenate((temp,X[i]),axis=1)
    X = temp
    
    enc = preprocessing.OneHotEncoder(n_values=n_values)
    enc.fit(X)
    X_byte = enc.transform(X).toarray()
    
    if (addAge):
        X_byte = concatenate((X_byte,userAge.age.values.reshape(1,len(userAge[['age']])).transpose()),axis=1)
    if (usePrints):
        print('Xlabel 0:', Xlabel.head(1))
        print('X 0:',X[0])
        print('n_values: ', n_values)
        print('X_byte 0:', X_byte[0])
        n_loc = 0
        for n in n_values:
            print(n_loc, ':  ', X_byte[0][(n_loc):(n_loc+n)] , '  -----end-----')
            n_loc += n
    
    X_byteDF = makeDataFrame(X_byte,labels,n_values,addAge)
    return X_byteDF

def makeDataFrame(X_byte,labels,n_values,addAge):
    columns = list(range(len(X_byte[0])))
    n_index = 0
    column_index = 0
    for label in labels :
        for index in range(n_values[n_index]) :
            columns[column_index] = label + '_' + str(index)
            column_index += 1
        n_index += 1
    if (addAge):
        columns[column_index] = 'age'
    X_byteDF = pd.DataFrame(data=X_byte, columns=columns)
    return X_byteDF

# Sort users who have age and who dosent have age
def sortAge(users):
    if ((users.columns == 'validAge').any()): # if users have validAge bit
        X_nan_age = users[users.validAge == 0].drop(['age', 'validAge'], 1)
        X_age = users[users.validAge == 1].drop('validAge', 1)
    else: # if users have dosent validAge bit
        X_nan_age = users[users.age == '-unknown-'].drop('age',1)
        X_age = users[users.age != '-unknown-']
    return X_nan_age, X_age

# Sort users who have gender and who dosent have gender
def sortGender(users):
    X_nan_gender = users[users.gender == '-unknown-'].drop('gender', 1)
    X_gender = users[users.gender != '-unknown-']
    return X_nan_gender, X_gender

# Sort users to 4 groups: yes/no gender/age
def sortAgeGender(users, usePrints=False):
    X_nan_age, X_age = sortAge(users)
    X_nan_age_nan_gender, X_nan_age_gender = sortGender(X_nan_age)
    X_age_nan_gender, X_age_gender = sortGender(X_age)
    if (usePrints):
        print('no age, no gender:',len(X_nan_age_nan_gender.index))
        print('no age, yes gender:',len(X_nan_age_gender.index))
        print('yes age, no gender:',len(X_age_nan_gender.index))
        print('yes age, yes gender:',len(X_age_gender.index))
    X={}
    X['nan_age_nan_gender'] = X_nan_age_nan_gender
    X['nan_age_gender'] = X_nan_age_gender
    X['age_nan_gender'] = X_age_nan_gender
    X['age_gender'] = X_age_gender
    return X

# Load the data into DataFrames
def loadAndUpdateAirBnbUsers(path, youngTreshold=15, oldTreshold=85, normalizeAge=True ,addValidAgeBit=True):
    users = pd.read_csv(path)
    users.gender.replace(np.nan,'-unknown-', inplace=True)
    users.first_affiliate_tracked.replace(np.nan,'-unknown-', inplace=True)
    #age:
    users.loc[users.age > oldTreshold, 'age'] = np.nan
    users.loc[users.age < youngTreshold, 'age'] = np.nan
    users.age.replace(np.nan,'-unknown-', inplace=True)
    if (addValidAgeBit):
        users['validAge'] = 0
        users.loc[users.age != '-unknown-', 'validAge'] = 1
        users.loc[users.age == '-unknown-', 'age'] = users[users.age != '-unknown-'].age.mean()  
    if normalizeAge : users.age = users.age / (oldTreshold + 1)
    #countries:
    users['NDF_US_OTHER_destination'] = 0
    users.loc[(users.country_destination == 'NDF'), 'NDF_US_OTHER_destination'] = 'NDF'
    users.loc[(users.country_destination == 'US'), 'NDF_US_OTHER_destination'] = 'US'
    users.loc[(users.country_destination != 'NDF') & (users.country_destination != 'US'), 'NDF_US_OTHER_destination'] = 'other'
    users['NDF_OTHER_destination'] = 0
    users.loc[(users.country_destination == 'NDF'), 'NDF_OTHER_destination'] = 'NDF'
    users.loc[(users.country_destination != 'NDF'), 'NDF_OTHER_destination'] = 'other'
    
    categorical_features = [
        'affiliate_channel',
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
        'signup_method'
    ]
    
    for categorical_feature in categorical_features:
        users[categorical_feature] = users[categorical_feature].astype('category')
    users['date_account_created'] = pd.to_datetime(users['date_account_created'])
    users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
    users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')
    users.first_affiliate_tracked.replace('NaN','-unknown-', inplace=True) 
    return users