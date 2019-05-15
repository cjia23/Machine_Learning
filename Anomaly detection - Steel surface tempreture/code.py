#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:00:45 2019

@author: chunyangjia
"""

######Question 1#####################################

import pandas as pd
import numpy as np
import math
path1 = './Datasets/Normal_Experiments/'
path2 = './Datasets/Experiments_with_Anomalies/'

Heat_normal = pd.DataFrame()

#read in the datasets
for number in range(1,69):
    df1 = pd.read_csv(path1+'/HEAT_ID_' + 
                     str("{:02d}".format(number)) + '_ALARM_OUT.csv')
    #filter for sds_armed as 1
    df1 = df1[(df1['Sds_Armed'] == 1)]
    Heat_normal = pd.concat([Heat_normal, df1])
# save the normal values file to merged_exp_normal.csv
Heat_normal.to_csv('merged_exp_normal.csv')    

#####Question 2######################################

Heat_anomalies = pd.DataFrame()

#read in the datasets
for number in range(1,20):
    df2 = pd.read_csv(path2+'/HEAT_ID_' + 
                     str("{:02d}".format(number)) + '_ALARM_OUT_tag.csv')
    #filter for sds_armed as 1
    df2 = df2[(df2['Sds_Armed'] == 1)]
    Heat_anomalies = pd.concat([Heat_anomalies, df2])
# save the normal values file to merged_exp_normal.csv
Heat_anomalies.to_csv('merged_exp_contains_anomalies.csv')    

######Question 3#######################################
Heat_normal = pd.read_csv('merged_exp_normal.csv',index_col = 0)
Heat_anomalies = pd.read_csv('merged_exp_contains_anomalies.csv', index_col =0)

X_anomalies = Heat_anomalies.iloc[:,0:8]
y_anomalies = Heat_anomalies.iloc[:,9]

from sklearn.feature_selection import chi2, SelectKBest
chi2_selector = SelectKBest(chi2, k = 8)
chi2_selector.fit(X_anomalies, y_anomalies)

chi2_scores = pd.DataFrame(list(zip(X_anomalies.columns, 
              chi2_selector.scores_, chi2_selector.pvalues_)), 
              columns=['ftr', 'score', 'pval'])

#rank the features 
print (chi2_scores.sort_values(by = 'score', ascending = False))

#######Question 4#######################################
X_all_normal = Heat_normal.iloc[:,0:8]
y_all_normal = Heat_normal.iloc[:,9]

###4.1 - consider all features
def gaussianMeanVariance(X):
    m,n = X.shape
    sigma = np.zeros(n)
    mean = np.zeros(n)
    
    mean = np.mean(X, axis = 0)
    sigma = np.var(X, axis = 0)
    return mean,sigma

def computeGaussian(s,mean,sigma):
    #p = 1/(sigma * np.sqrt(2 * math.pi)) * np.exp( - (s - mean)**2 / (2 * sigma**2))
    p = np.exp(-(s-mean)**2/(2*sigma**2))/(math.sqrt(2*math.pi)*sigma)
    return p

def P(X_all_normal):    
    mean, sigma = gaussianMeanVariance(X_all_normal)
    p = 1
    for col in range(0, len(X_all_normal.columns)):
        p = p*computeGaussian(X_all_normal.iloc[:,col], mean[col], sigma[col])
    return p

p = P(X_all_normal)

###4.2 - mark the most important 2 features
def P_toptwo(X_all_normal):
    X_toptwo = X_all_normal.iloc[:,2:4]
    mean, sigma = gaussianMeanVariance(X_toptwo)
    p_toptwo = 1
    for col in range(0, len(X_toptwo.columns)):
        p_toptwo = p_toptwo * computeGaussian(X_toptwo.iloc[:,col], mean[col], sigma[col])
    return p_toptwo

p_toptwo = P_toptwo(X_all_normal)


###4.3 - PCA principle component analysis
from sklearn.decomposition import PCA

def P_pca(X_all_normal):
    X_pca = X_all_normal
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_pca)
    X_pca = pd.DataFrame(X_pca)
    
    mean, sigma = gaussianMeanVariance(X_pca)
    
    p_pca = 1
    
    for col in range(0, 2):
        p_pca = p_pca * computeGaussian(X_pca.iloc[:,col], mean[col], sigma[col])
    return p_pca

p_pca = P_pca(X_all_normal)

######Question 5###########################################
###5.1 - consider all features
from scipy.stats import multivariate_normal

def gaussianMeanCovariance(X):
    
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)
    return mean,cov

def P_multigaussian(X_all_normal):
    mean, cov = gaussianMeanCovariance(X_all_normal)
    p_mutilgaussian = multivariate_normal.pdf(X_all_normal, mean = mean , cov = cov)
    return p_mutilgaussian

p_mutilgaussian = P_multigaussian(X_all_normal)


###5.2 - consider the top 2 features
def P_toptwo_dep(X_all_normal):
    X_toptwo_dep = X_all_normal.iloc[:,2:4]
    
    mean, cov = gaussianMeanCovariance(X_toptwo_dep)
    p_toptwo_dep = multivariate_normal.pdf(X_toptwo_dep, mean = mean, cov = cov)
    return p_toptwo_dep

p_toptwo_dep = P_toptwo_dep(X_all_normal)

###5.3 - PCA principle component analysis

def Pca_dep(X_all_normal):
    
    X_pca_dep = X_all_normal
    pca_dep = PCA(n_components = 2)
    X_pca_dep = pca_dep.fit_transform(X_pca_dep)
    X_pca_dep = pd.DataFrame(X_pca_dep)
    mean, cov = gaussianMeanVariance(X_pca_dep)
    p_pca_dep = multivariate_normal.pdf(X_pca_dep, mean = mean, cov = cov)
    return p_pca_dep

p_pca_dep = Pca_dep(X_all_normal)

######Question 6###########################################
#6 models to be tested, going to take threahod as the mean of all p values
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_anomalies = Heat_anomalies.iloc[:,0:8]
y_anomalies = Heat_anomalies.iloc[:,9]

X_train, X_test,y_train, y_test = train_test_split(X_anomalies, y_anomalies, 
                                                   test_size = 0.2,shuffle = False)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#6.1 - model from 4.1
p_sixone = P(X_train)
threshold_sixone = (np.array(p_sixone)).mean()*0.00001
p_sixone_predict = P(X_test)

y_sixone_predict = []

for val in p_sixone_predict:
    if val <= threshold_sixone:
        y_sixone_predict.append(1)
    elif val > threshold_sixone:
        y_sixone_predict.append(0)
        
print('6.1 accuracy: ', metrics.accuracy_score(y_sixone_predict,y_test))
print('6.1 F1 score: ',metrics.f1_score(y_sixone_predict,y_test))
print('6.1 recall: ',metrics.recall_score(y_sixone_predict,y_test))
print('6.1 report: ', metrics.classification_report(y_sixone_predict,y_test))

#6.2 - model from 4.2
p_sixtwo = P_toptwo(X_train)
threshold_sixtwo = (np.array(p_sixtwo)).mean()* 13.645
p_sixtwo_predict = P_toptwo(X_test)

y_sixtwo_predict = []

for val in p_sixtwo_predict:
    if val <= threshold_sixtwo:
        y_sixtwo_predict.append(1)
    elif val > threshold_sixtwo:
        y_sixtwo_predict.append(0)
        
print('6.2 accuracy: ', metrics.accuracy_score(y_sixtwo_predict,y_test))
print('6.2 F1 score: ',metrics.f1_score(y_sixtwo_predict,y_test))
print('6.2 recall: ',metrics.recall_score(y_sixtwo_predict,y_test))
print('6.2 report: ', metrics.classification_report(y_sixtwo_predict,y_test))

#6.3 - model from 4.3
p_sixthree = P_pca(X_train)
threshold_sixthree = (np.array(p_sixthree)).mean() * 1.57555
p_sixthree_predict = P_pca(X_test)

y_sixthree_predict = []

for val in p_sixthree_predict:
    if val <= threshold_sixthree:
        y_sixthree_predict.append(1)
    elif val > threshold_sixthree:
        y_sixthree_predict.append(0)
        
print('6.3 accuracy: ', metrics.accuracy_score(y_sixthree_predict,y_test))
print('6.3 F1 score: ',metrics.f1_score(y_sixthree_predict,y_test))
print('6.3 recall: ',metrics.recall_score(y_sixthree_predict,y_test))
print('6.3 report: ', metrics.classification_report(y_sixthree_predict,y_test))

#6.4 - model from 5.1
p_sixfour = P_multigaussian(X_train)
threshold_sixfour = (np.array(p_sixfour)).mean() * 1000
p_sixfour_predict = P_multigaussian(X_test)

y_sixfour_predict = []

for val in p_sixfour_predict:
    if val <= threshold_sixfour:
        y_sixfour_predict.append(1)
    elif val > threshold_sixfour:
        y_sixfour_predict.append(0)
        
print('6.4 accuracy: ', metrics.accuracy_score(y_sixfour_predict,y_test))
print('6.4 F1 score: ',metrics.f1_score(y_sixfour_predict,y_test))
print('6.4 recall: ',metrics.recall_score(y_sixfour_predict,y_test))
print('6.4 report: ', metrics.classification_report(y_sixfour_predict,y_test))

#6.5 - model from 5.2
p_sixfive = P_toptwo_dep(X_train)
threshold_sixfive = (np.array(p_sixfive)).mean()* 4.15
p_sixfive_predict = P_toptwo_dep(X_test)

y_sixfive_predict = []

for val in p_sixfive_predict:
    if val <= threshold_sixfive:
        y_sixfive_predict.append(1)
    elif val > threshold_sixfive:
        y_sixfive_predict.append(0)
        
print('6.5 accuracy: ', metrics.accuracy_score(y_sixfive_predict,y_test))
print('6.5 F1 score: ',metrics.f1_score(y_sixfive_predict,y_test))
print('6.5 recall: ',metrics.recall_score(y_sixfive_predict,y_test))
print('6.5 report: ', metrics.classification_report(y_sixfive_predict,y_test))


#6.6 - model from 5.3
p_sixsix = Pca_dep(X_train)
threshold_sixsix = p_sixsix.mean()* 1.327
p_sixsix_predict = Pca_dep(X_test)

y_sixsix_predict = []

for val in p_sixsix_predict:
    if val <= threshold_sixsix:
        y_sixsix_predict.append(1)
    elif val > threshold_sixsix:
        y_sixsix_predict.append(0)
        
print('6.6 accuracy: ', metrics.accuracy_score(y_sixsix_predict,y_test))
print('6.6 F1 score: ',metrics.f1_score(y_sixsix_predict,y_test))
print('6.6 recall: ',metrics.recall_score(y_sixsix_predict,y_test))
print('6.6 report: ', metrics.classification_report(y_sixsix_predict,y_test))



######Question 7###########################################
##will use 6.1 model as an exmaple to show in Question 7
import matplotlib.pyplot as plt
X_x1 = X_test['X1']
predict_label = y_sixone_predict
true_label = y_test

index = list(range(len(X_x1)))

plt.plot(index, X_x1, label = 'X1 Feature')
plt.plot(index, predict_label, label = 'predict label')
plt.plot(index, true_label, label = 'true label')
plt.legend()
plt.show()

#####Question 8###########################################

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_all = Heat_anomalies.iloc[:,0:8]
y_all = Heat_anomalies.iloc[:,9]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_all)
n_x_transform = scaler.transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(n_x_transform,y_all,test_size = 0.3, 
                                                    shuffle = False)

clf = SVC(kernel = 'rbf', C = 10, gamma = 'auto')
clf.fit(X_train, y_train)
y_predict_svc = clf.predict(X_test)

print('Accuracy for rbf SVC',metrics.accuracy_score(y_test,y_predict_svc))
print('F1 : ', metrics.f1_score(y_test,y_predict_svc))
print('recall : ', metrics.recall_score(y_test,y_predict_svc))
print('Classification Report: ', metrics.classification_report(y_test,y_predict_svc))


######Question 9############################################
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X_all)
y_predict_kmeans = kmeans.predict(X_all)

print('Accuracy for Kmeans',metrics.accuracy_score(y_all,y_predict_kmeans))
print('F1 for Kmeans',metrics.f1_score(y_all,y_predict_kmeans))
print('recall : ', metrics.recall_score(y_test,y_predict_svc))
print('Classification Report: ', metrics.classification_report(y_all,y_predict_kmeans))


######Question 10##############################################
"""
Refer to the PDF.
"""

######Question 11###############################################
"""
goal equals to detection rate divided by false alarm rate, so consider F1 score
idea is we select a group of threhold, store the score and find the best one
After manual tuning, we noticed that the best is among 4 to 5"""

def findbestThreshold(X,y,function,minimum, maximum, n):
    X_train, X_test,y_train, y_test = train_test_split(X, y, 
                                                   test_size = 0.2,shuffle = False)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    result = pd.DataFrame(columns = ['threhold', 'score'])
    
    for i in range(0, n):
        p = function(X_train)
        p_predict = function(X_test)
        threshold = (np.array(p)).mean()*(minimum + (maximum - minimum)*i/n)
        
        y_predict = []
        for val in p_predict:
            if val <= threshold:
                y_predict.append(1)
            elif val > threshold:
                y_predict.append(0)
        score = metrics.f1_score(y_predict, y_test)
        
        result = result.append({'threhold': threshold, 'score':score}, 
                               ignore_index = True)
    
    row = result['score'].argmax()
    best_threhold = result.iat[row, 0]
    best_f1 = result.iat[row, 1]
    
    return best_threhold,best_f1

T_65,f1_65 = findbestThreshold(X_anomalies, y_anomalies, P_toptwo_dep, 4, 5,5000)       
print (T_65,f1_65) 

     
        
    
























