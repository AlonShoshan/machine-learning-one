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
import basicLib.featureUpdate as featUp
import re
import random

###########################################################
# Load the data into DataFrames                           #
# The feature struct is a list containing feature classes #
###########################################################
def loadAndUpdateFeatures(pathTrain, pathTest = np.nan ,featureStruct=featUp.generateFeatureStruct()):
    users = pd.read_csv(pathTrain)
    yRes = pd.DataFrame(users['country_destination'])
    users = users.drop('country_destination',axis=1)
    if pd.notnull(pathTest):
        usersTest = pd.read_csv(pathTest)
        users = pd.concat([users,usersTest],ignore_index=True)
    for feature in featureStruct:
        feature.update(users)
    yFeat = featUp.countriesFeature()
    yFeat.update(yRes)
    return users, yRes

    
#######################################################
# Fit predict test acording to predictMethod          #
# users provided by using the loadAndUpdate function  #
# featurelist provided by using the featureList class #
#######################################################    

def runAll(pathTrain, pathTest, category):
    users, yRes, usersTrain, usersTest = loadAndUpdate(pathTrain, pathTest)
    featureList1 = featureList(usersCol=users.columns)
    resUser, myLabels = fitPredict(users,yRes,featureList1.get(),category)
    resUser = cleanTest(usersTrain,usersTest,resUser,myLabels)
    resUser['id'] = users['id']
    return resUser, yRes

def loadAndUpdate(pathTrain, pathTest,featureStruct=featUp.generateFeatureStruct()):
    usersTrain = pd.read_csv(pathTrain)
    yRes = pd.DataFrame(usersTrain['country_destination'])
    usersTrain = usersTrain.drop('country_destination',axis=1)
    usersTest = pd.read_csv(pathTest)
    users = pd.concat([usersTrain,usersTest],ignore_index=True)
    for feature in featureStruct:
        feature.update(users)
    yFeat = featUp.countriesFeature()
    yFeat.update(yRes)
    return users, yRes, usersTrain, usersTest

def fitPredict(orgUsers,yRes,featureList,usePrints=False,random_state=1,normalize=True,crossPredict=False):
    users = orgUsers.copy(deep=True)
    initiateUsers(users)
    X = sortFeatures(users,featureList)
    X_byteDF,myLabels = category2bin(X, usePrints=usePrints)
    if normalize: normDf(X_byteDF)
    return X_byteDF, myLabels    
    
def cleanTest(usersTrain,usersTest,users,myLabels):
    myDrop = {}
    myList=('affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
            'first_browser', 'first_device_type', 'gender', 'language',
            'signup_app', 'signup_method', 'signup_flow')
    for i in myList:
        attrTest = getattr(usersTest,i)
        attrTrain = getattr(usersTrain,i)
        myDrop[i] = []
        for x in attrTest.unique():
            if x not in attrTrain.unique():
                myDrop[i].append(x)
                print(i,':', x)
                
    for key in myDrop.keys():
        for testCat in myDrop[key]:
            print(key,',', testCat)
            if pd.isnull(testCat): continue
            myInd = myLabels[key].classes_.tolist().index(testCat)
            label = key + '_' + str(myInd)
            users = users.drop(label,axis=1)
            
    users = users.drop('date_account_created_month_6',axis=1)
    users = users.drop('date_account_created_month_7',axis=1)
    users = users.drop('date_account_created_month_8',axis=1)

    return users

def category2bin(users, usePrints=False):
    Xlabel = users
    nonCategoryList=[]
    for feature, dtype in Xlabel.dtypes.iteritems():
        if dtype.name != 'category':
            nonCategoryList.append(feature)
    nonCategoryDF = Xlabel[nonCategoryList]
    Xlabel = Xlabel.drop(nonCategoryList, 1)

    labels = Xlabel.columns
    myLabels = {} # holds the encode labels with value between 0 and n-1
    n_values = [] # holds the length of each feature in bits ('gender' with 4 option will be 4 bits)
    X = []
    for label in Xlabel:
        labelAttr = getattr(Xlabel, label)
        myLabels[label] = preprocessing.LabelEncoder()
        myLabels[label].fit(labelAttr)
        if size(labelAttr.unique()) > 400:
            raise TypeError('The feature:', label, 'is too big to handle as binary')
        n_values.append(size(labelAttr.unique()))
        #transform: turn options into 0 to n-1,  reshape+transpose: align the coulumn so it be easier to add them to the table 
        X.append(myLabels[label].transform(labelAttr).reshape(1,len(labelAttr)).transpose())
    
    #There is no category to change
    if len(X) == 0: return users
    temp = np.array([]).reshape(len(X[0]),0)
    for i in range(len(X)):
        temp = concatenate((temp,X[i]),axis=1)
    X = temp
    
    enc = preprocessing.OneHotEncoder(n_values=n_values)
    enc.fit(X)
    X_byte = enc.transform(X).toarray()
    
     
    if (usePrints):
        print('Xlabel 0:', Xlabel.head(1))
        print('X 0:',X[0])
        print('n_values: ', n_values)
        print('X_byte 0:', X_byte[0])
        n_loc = 0
        for n in n_values:
            print(n_loc, ':  ', X_byte[0][(n_loc):(n_loc+n)] , '  -----end-----')
            n_loc += n
    
    X_byteDF = makeDataFrame(X_byte,labels,n_values)
    result = pd.concat([X_byteDF, nonCategoryDF], axis=1)
    return result, myLabels
#######################################################
# Fit predict test acording to predictMethod          #
# users provided by using the loadAndUpdate function  #
# featurelist provided by using the featureList class #
#######################################################
def fitPredictAndTest(orgUsers,featureList,category,predictMethod,usePrints=False,random_state=1,normalize=True,crossPredict=False):
    users = orgUsers.copy(deep=True)
    initiateUsers(users)
    X = sortFeatures(users,featureList)
    X_byteDF = category2binaryFeatures(X, usePrints=usePrints)
    y = sortResults(users,category)
    if normalize: normDf(X_byteDF)
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

def getXbyte(orgUsers,featureList,category,normalize=True, usePrints=False):
    users = orgUsers.copy(deep=True)
    initiateUsers(users)
    X = sortFeatures(users,featureList)
    X_byteDF = category2binaryFeatures(X, usePrints=usePrints)
    y = sortResults(users,category)
    if normalize: normDf(X_byteDF)
    return X_byteDF, y

# normolize the given dataFrame in the given values
def normDf(myDF, minNum = -1, maxNum = 1):
    myRange = maxNum - minNum
    for column in myDF.columns:
        myDF[column] = myRange * (myDF[column]-myDF[column].min())/(myDF[column].max()-myDF[column].min()) + minNum    
    
# sort wanted Features
def sortFeatures(users,featureList):
    return users[featureList]
# sort wanted Results
def sortResults(users,category):
    return users[category]
# initiate the users
def initiateUsers(users,usePrints=False):
    dateFeaturs(users,Month=True,DayOfWeek=True)

# add date related features
def dateFeaturs(users,Month=False,DayOfWeek=False,DayOfYear=False,WeekofYear=False,Quarter=False,account_created=True,first_active=True):
    if account_created :
        add_date('date_account_created',users,Month=Month,DayOfWeek=DayOfWeek,
                 DayOfYear=DayOfYear,WeekofYear=WeekofYear,Quarter=Quarter)
    if first_active    : 
        add_date('date_first_active',users,Month=Month,DayOfWeek=DayOfWeek,
                 DayOfYear=DayOfYear,WeekofYear=WeekofYear,Quarter=Quarter)

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
    if Month : users[dateFeature + '_month'] = pd.Series(month).astype('category')
    if DayOfWeek : users[dateFeature + '_dayofweek'] = pd.Series(dayofweek).astype('category')
    if DayOfYear : users[dateFeature + '_dayofyear'] = pd.Series(dayofyear).astype('category')
    if WeekofYear : users[dateFeature + '_weekofyear'] = pd.Series(weekofyear).astype('category')
    if Quarter : users[dateFeature + '_quarter'] = pd.Series(quarter).astype('category')


#####################################################
# Transforme categoryal features to binary features #
# The given feature should be defined to 'category' #
# or anything else (int, float, etc.). it will      #
# transform only the category and will exit if the  #
# amount of options is too big.                     #
#####################################################
def category2binaryFeatures(users, usePrints=False):
    Xlabel = users
    nonCategoryList=[]
    for feature, dtype in Xlabel.dtypes.iteritems():
        if dtype.name != 'category':
            nonCategoryList.append(feature)
    nonCategoryDF = Xlabel[nonCategoryList]
    Xlabel = Xlabel.drop(nonCategoryList, 1)

    labels = Xlabel.columns
    myLabels = {} # holds the encode labels with value between 0 and n-1
    n_values = [] # holds the length of each feature in bits ('gender' with 4 option will be 4 bits)
    X = []
    for label in Xlabel:
        labelAttr = getattr(Xlabel, label)
        myLabels[label] = preprocessing.LabelEncoder()
        myLabels[label].fit(labelAttr)
        if size(labelAttr.unique()) > 400:
            raise TypeError('The feature:', label, 'is too big to handle as binary')
        n_values.append(size(labelAttr.unique()))
        #transform: turn options into 0 to n-1,  reshape+transpose: align the coulumn so it be easier to add them to the table 
        X.append(myLabels[label].transform(labelAttr).reshape(1,len(labelAttr)).transpose())
    
    #There is no category to change
    if len(X) == 0: return users
    temp = np.array([]).reshape(len(X[0]),0)
    for i in range(len(X)):
        temp = concatenate((temp,X[i]),axis=1)
    X = temp
    
    enc = preprocessing.OneHotEncoder(n_values=n_values)
    enc.fit(X)
    X_byte = enc.transform(X).toarray()
    
     
    if (usePrints):
        print('Xlabel 0:', Xlabel.head(1))
        print('X 0:',X[0])
        print('n_values: ', n_values)
        print('X_byte 0:', X_byte[0])
        n_loc = 0
        for n in n_values:
            print(n_loc, ':  ', X_byte[0][(n_loc):(n_loc+n)] , '  -----end-----')
            n_loc += n
    
    X_byteDF = makeDataFrame(X_byte,labels,n_values)
    result = pd.concat([X_byteDF, nonCategoryDF], axis=1)
    return result

# Combine the category features with the non-category into a dataframe
def makeDataFrame(X_byte,labels,n_values):
    columns = list(range(len(X_byte[0])))
    n_index = 0
    column_index = 0
    for label in labels :
        for index in range(n_values[n_index]) :
            columns[column_index] = label + '_' + str(index)
            column_index += 1
        n_index += 1
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
    if ((users.columns == 'validGender').any()): # if users have validGender bit
        X_nan_gender = users[users.validGender == 0].drop(['gender', 'validGender'], 1)
        X_gender = users[users.validGender == 1].drop('validGender', 1)
    else: # if users have dosent validGender bit
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


##remove columns from a dataframe that match the regex in the list
def removeColumns(users, myList):
    for category in myList:
        for i in users.columns:
            if re.match(r'\b' + category + r'(.*)',i):
                users = users.drop(i,axis=1)
    return users
    
##remove columns from a dataframe that match the regex in the list
def keepColumns(users, myList):
    myListBig = []
    for category in myList:
        for i in users.columns:
            if re.match(r'\b' + category + r'(.*)',i):
               myListBig.append(i) 
    return users[myListBig]

############################################################
# Class that hold the featurelist - generate a default one #
# All methods only accept lists (meaning: [])              #
############################################################
class featureList():
    def __init__(self, myList = ('affiliate_channel','affiliate_provider','first_affiliate_tracked',
                           'first_browser','first_device_type','language','signup_app','signup_method',
                           'signup_flow','date_account_created_month',
                           'date_account_created_dayofweek','age',
                           'validAge','gender'), 
                 regexList = ("action_", "num_of_devices","total_time","timeAct_","dev_"),
                 usersCol = ()):
        self.list = list(myList)
        self.addByRegex(regexList, usersCol)
    def add(self, myList):
        for i in myList:
            self.list.append(i)
    def addByRegex(self, regexList, usersCol):
        for i in usersCol:
            for regex in regexList:    
                if re.match(r'\b' + regex + r'(.*)',i):
                    self.list.append(i)
    def remove(self, myList):
        for i in myList:
            self.list.remove(i)
    def removeByRegex(self, regexList):
        for regex in regexList:
            self.list = [x for x in self.list if not re.match(r'(.*)' + regex + r'(.*)',x)]
    def get(self):
        return self.list
    @staticmethod
    def randomList(myList, minFeaturs=4):
        numOfFeature = len(myList)
        numOfFeautersInNewModel = random.randint(minFeaturs, numOfFeature)
        newFeatureListIndexes = random.sample(range(0, numOfFeature), numOfFeautersInNewModel)
        newFeatureList = [myList[index] for index in newFeatureListIndexes]
        return newFeatureList
    
    
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