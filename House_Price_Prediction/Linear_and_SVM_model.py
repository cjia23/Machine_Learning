#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:01:58 2018

@author: chunyangjia
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#bring in the data and make the id into index
train_path = './housing_price/train.csv'
df_data = pd.read_csv(train_path)
print (df_data.head())
df_corr = df_data.corr()
#########################################################

df_data.index = df_data['Id']
df_data = df_data[['LotFrontage','LotArea','OverallQual','OverallCond',
              'YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','2ndFlrSF',
              'GrLivArea','FullBath','BedroomAbvGr','KitchenAbvGr',
              'TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF',
              'PoolArea','YrSold','SalePrice']]
df_data.fillna(0,inplace = True)
print (df_data.head())

for i in range (1,1460):
    if df_data['SalePrice'][i] > 500000:
        print(2)
        df_data.drop(i, inplace = True)
        
#training and testing sets split
X = df_data.loc[:,'LotFrontage':'YrSold']
y = df_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,shuffle = True)

#########################################################
#Linear Regression

from sklearn.linear_model import LinearRegression
Model_LR = LinearRegression()
Model_LR.fit(X_train,y_train)

y_LR_pred = Model_LR.predict(X_test)

from sklearn import metrics

print("RMSE for linear regression: ", np.sqrt(metrics.mean_squared_error(y_test,y_LR_pred)))
print("MAE for linear regression: ", metrics.mean_absolute_error(y_test,y_LR_pred))
#df_linear = pd.DataFrame({'Actual':y_test, 'Linear Regression Predicted':y_pred})

#print(df_linear)
###########################################################

from sklearn.svm import SVR

SVM_model = SVR(kernel = 'rbf')
SVM_model.fit(X_train,y_train)

y_svm_pred = SVM_model.predict(X_test)

print("RMSE for SVR: ", np.sqrt(metrics.mean_squared_error(y_test,y_svm_pred)))
print("MAE for SVR: ", metrics.mean_absolute_error(y_test,y_svm_pred))
df_SVM = pd.DataFrame({'Actual':y_test, 'SVM_Predicted':y_svm_pred,'Linear Regression Predicted':y_LR_pred})













