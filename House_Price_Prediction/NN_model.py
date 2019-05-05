#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:28:53 2018

@author: chunyangjia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 22 14:01:58 2018

@author: chunyangjia
"""

#1. Read in the data and check correlation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_path = './housing_price/train.csv'
df_data = pd.read_csv(train_path)
print(df_data.shape)
print(df_data['SalePrice'].describe())
sns.distplot(df_data['SalePrice'])
plt.show()

#2. Feature Selection and pre-data modification
df_corr = df_data.corr()
df_data.index = df_data['Id']
df_data = df_data[['LotFrontage','LotArea','OverallQual','OverallCond',
              'YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','2ndFlrSF',
              'GrLivArea','FullBath','BedroomAbvGr','KitchenAbvGr',
              'TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF',
              'PoolArea','YrSold','SalePrice']]
df_data.fillna(0,inplace = True)

#3. Removing outliers and filling Nan values
df_data.fillna(0,inplace = True)
for i in range (1,1460):
    if df_data['SalePrice'][i] > 500000:
        print(2)
        df_data.drop(i, inplace = True)


#4. training and testing sets split
X = df_data.loc[:,'LotFrontage':'YrSold']
y = df_data['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,shuffle = False)

#5.Data preprocessing
from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)


#6.Model Training and Tuning
from keras import models
from keras import layers
from keras import losses

#Model_1 training
Model_1 = models.Sequential()
Model_1.add(layers.Dense(19, input_dim= 19))
for i in range(0,10):
    Model_1.add(layers.Dense(60, activation = 'relu'))
Model_1.add(layers.Dense(1))
Model_1.compile(optimizer = 'adam', loss = losses.mean_squared_error)
Model_1.summary()
history1 = Model_1.fit(X_train, y_train,validation_split = 0.1,
                    epochs = 50, batch_size = 4)
y1_predict = Model_1.predict(X_test)
Model_1.save('Model_1.h5')

#Model_2 trainig
Model_2 = models.Sequential()
Model_2.add(layers.Dense(19, input_dim= 19))
for i in range(0,15):
    Model_2.add(layers.Dense(60, activation = 'relu'))
Model_2.add(layers.Dense(1))
Model_2.compile(optimizer = 'adam', loss = losses.mean_squared_error)
Model_2.summary()
history2 = Model_2.fit(X_train, y_train,validation_split = 0.1,
                    epochs = 50, batch_size = 4)
y2_predict = Model_2.predict(X_test)
Model_2.save('Model_2.h5')

#Model_3 trainig
Model_3 = models.Sequential()
Model_3.add(layers.Dense(19, input_dim= 19))
for i in range(0,10):
    Model_3.add(layers.Dense(90, activation = 'relu'))
Model_3.add(layers.Dense(1))
Model_3.compile(optimizer = 'adam', loss = losses.mean_squared_error)
Model_3.summary()
history3 = Model_3.fit(X_train, y_train,validation_split = 0.1,
                    epochs = 50, batch_size = 4)
y3_predict = Model_3.predict(X_test)
Model_3.save('Model_3.h5')

#Model_4 trainig
Model_4 = models.Sequential()
Model_4.add(layers.Dense(19, input_dim= 19))
for i in range(0,10):
    Model_4.add(layers.Dense(60, activation = 'tanh'))
Model_4.add(layers.Dense(1))
Model_4.compile(optimizer = 'adam', loss = losses.mean_squared_error)
Model_4.summary()
history4 = Model_4.fit(X_train, y_train,validation_split = 0.1,
                    epochs = 50, batch_size = 4)
y4_predict = Model_4.predict(X_test)
Model_4.save('Model_4.h5')

#Model_5 trainig
Model_5 = models.Sequential()
Model_5.add(layers.Dense(19, input_dim= 19))
for i in range(0,10):
    Model_5.add(layers.Dense(60, activation = 'relu'))
Model_5.add(layers.Dense(1))
Model_5.compile(optimizer = 'adam', loss = losses.mean_absolute_error)
Model_5.summary()
history5 = Model_5.fit(X_train, y_train,validation_split = 0.1,
                    epochs = 50, batch_size = 8)
y5_predict = Model_5.predict(X_test)
Model_5.save('Model_5.h5')



from sklearn import metrics
print("RMSE for NN Model 1: ", np.sqrt(metrics.mean_squared_error(y_test,y1_predict)))
print("MAE for NN Model 1: ", metrics.mean_absolute_error(y_test,y1_predict))

print("RMSE for NN Model 2: ", np.sqrt(metrics.mean_squared_error(y_test,y2_predict)))
print("MAE for NN Model 2: ", metrics.mean_absolute_error(y_test,y2_predict))

print("RMSE for NN Model 3: ", np.sqrt(metrics.mean_squared_error(y_test,y3_predict)))
print("MAE for NN Model 3: ", metrics.mean_absolute_error(y_test,y3_predict))

print("RMSE for NN Model 4: ", np.sqrt(metrics.mean_squared_error(y_test,y4_predict)))
print("MAE for NN Model 4: ", metrics.mean_absolute_error(y_test,y4_predict))

print("RMSE for NN Model 5: ", np.sqrt(metrics.mean_squared_error(y_test,y5_predict)))
print("MAE for NN Model 5: ", metrics.mean_absolute_error(y_test,y5_predict))

plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model_1 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
plt.style.use('ggplot')
plt.plot(y1_predict,y_test,'ro')
plt.xlabel('Model_1 Predictions', fontsize = 10)
plt.ylabel('Reality', fontsize = 10)
plt.title('Model 1 Predictions VS Reality', fontsize = 10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model_2 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
plt.style.use('ggplot')
plt.plot(y2_predict,y_test,'ro')
plt.xlabel('Model_2 Predictions', fontsize = 10)
plt.ylabel('Reality', fontsize = 10)
plt.title('Model 2 Predictions VS Reality', fontsize = 10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('Model_3 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
plt.style.use('ggplot')
plt.plot(y3_predict,y_test,'ro')
plt.xlabel('Model_3 Predictions', fontsize = 10)
plt.ylabel('Reality', fontsize = 10)
plt.title('Model 3 Predictions VS Reality', fontsize = 10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

plt.plot(history4.history['loss'])
plt.plot(history4.history['val_loss'])
plt.title('Model_4 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
plt.style.use('ggplot')
plt.plot(y4_predict,y_test,'ro')
plt.xlabel('Model_4 Predictions', fontsize = 10)
plt.ylabel('Reality', fontsize = 10)
plt.title('Model 4 Predictions VS Reality', fontsize = 10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('Model_5 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper right')
plt.show()

fig, ax = plt.subplots(figsize = (5,5))
plt.style.use('ggplot')
plt.plot(y5_predict,y_test,'ro')
plt.xlabel('Model_5 Predictions', fontsize = 10)
plt.ylabel('Reality', fontsize = 10)
plt.title('Model 5 Predictions VS Reality', fontsize = 10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()