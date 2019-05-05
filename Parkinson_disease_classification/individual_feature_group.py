#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:32:46 2019

@author: chunyangjia
"""

######import libraries#########################################
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression #导入逻辑回归库
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#read in datasets
data = pd.read_csv("./pd_speech_features.csv")
print(data)
pd.isnull(data)

######initialize the list to store final results
name = input("please enter the feature group name: (baseline,mfcc,wavelet,vocal_fold,time_frequency,tqwt)")

if name == 'baseline':
    result_baseline=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })
elif name == 'mfcc':
    result_mfcc=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })
elif name == 'wavelet':
    result_wavelet=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })
elif name == 'vocal_fold':
    result_vocal_fold=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })
elif name == 'time_frequency':
    result_time_frequency=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })
elif name == 'tqwt':
    result_tqwt=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logistic regression','ensemble voting','ensemble stacking'],
                                'accuracy':['1','2','3','4','5',0,0],
                                'F1 score':['1','2','3','4','5',0,0],
                                'recall score':['1','2','3','4','5',0,0]
                            })

######split features and labels
x=data.iloc[1:757,0:754]
y=data.iloc[1:757,754]
x.rename(columns={ x.columns[0]: "id" ,x.columns[1]: "gender"}, inplace=True)
   
#######feature selection
####slice individual feature groups
id_gender=x.iloc[:,1:2]
baseline = x.iloc[:,3:23]
time_frequency = x.iloc[:,23:34]
vocal_fold = x.iloc[:,34:34+22]
mfcc = x.iloc[:,56:56+84]
wavelet_transform = x.iloc[:,140:140+182]
tqwt =x.iloc[:,322:754]

#####concat different feature groups
x_baseline=pd.concat([id_gender,baseline],axis=1 )
x_mfcc = pd.concat([id_gender,mfcc],axis=1 )
x_wavelet = pd.concat([id_gender,wavelet_transform],axis=1 )
x_vocal_fold = pd.concat([id_gender,vocal_fold],axis=1 )
x_time_frequency = pd.concat([id_gender,time_frequency],axis=1 )
x_tqwt = pd.concat([id_gender,tqwt],axis=1 )

######  select feature groups that you want to use.we have two methods for feature selection,
#####one is through choosing different feature groups(option includes x_baseline,
######  x_mfcc, x_wavelet, x_vocal_fold ,x_time_frequency), and x_out_mfcc and x_out_twqt are for rfe.
if name == 'baseline':
    x_use=x_baseline
elif name == 'mfcc':
    x_use = x_mfcc
elif name == 'vocal_fold':
    x_use = x_vocal_fold
elif name == 'wavelet':
    x_use = x_wavelet
elif name == 'time_frequency':
    x_use = x_time_frequency
elif name == 'tqwt':
    x_use = x_tqwt
    
#####split dataset
#####train: validation: test=6:2:2
"""
   1. In this section, we split the data manually into three portions, train
      test and validation. We didn't split it randomly because each participant 
      has been recorded three times, there will be artificial overlap.
"""
x_train=x_use.iloc[0:454,:]
y_train=y[0:454]
x_validation=x_use.iloc[454:605,:]
y_validation=y[454:605]
x_test=x_use.iloc[605:757,:]
y_test=y[605:757]

scaler= preprocessing.StandardScaler().fit(x_train)
n_x_train=scaler.transform(x_train)
n_x_validation=scaler.transform(x_validation)
n_x_test=scaler.transform(x_test)

#######classification
"""
   In this section, we tune the parameters for svm linear and rbf kernel as well
   as random forests.
"""
if name == 'baseline':
    #features for svm linear
    C_linear = 10
    #features for svm rbf
    C_rbf = 10
    gamma = 0.01
    #features for random forest
    random_state = None
    n_estimators=2000
    max_features='auto'
    oob_score = True
elif name == 'mfcc':
    #features for svm linear
    C_linear = 0.1
    #features for svm rbf
    C_rbf = 1
    gamma = 0.01
    #features for random forest
    random_state = 48
    n_estimators= 500
    max_features='auto'
    oob_score = True
elif name == 'vocal_fold':
    #features for svm linear
    C_linear = 0.1
    #features for svm rbf
    C_rbf = 10
    gamma = 0.01
    #features for random forest
    random_state = None
    n_estimators= 1000
    max_features='auto'
    oob_score = True
elif name == 'wavelet':
    #features for svm linear
    C_linear = 0.1
    #features for svm rbf
    C_rbf = 10
    gamma = 0.01
    #features for random forest
    random_state = 48
    n_estimators= 800
    max_features='auto'
    oob_score = True    
elif name == 'time_frequency':
    #features for svm linear
    C_linear = 0.1
    #features for svm rbf
    C_rbf = 100
    gamma = 0.01
    #features for random forest
    random_state = None
    n_estimators= 1000
    max_features='auto'
    oob_score = True    
elif name == 'tqwt':
    #features for svm linear
    C_linear = 0.1
    #features for svm rbf
    C_rbf = 10
    gamma = 0.0001
    #features for random forest
    random_state = None
    n_estimators= 1000
    max_features='auto'
    oob_score = True         


########svm(linear)
svm_linear = SVC(kernel='linear',C=C_linear)
svm_linear.fit(n_x_train,y_train)
svm_linear_predict_labels_validation = svm_linear.predict(n_x_validation)# print accuracy of training data
svm_linear_predict_labels = svm_linear.predict(n_x_test)# print accuracy of training data
print(svm_linear_predict_labels)

######svm metrics
######validation
svm_linear_acc_validation=accuracy_score(y_validation, svm_linear_predict_labels_validation )
svm_linear_f1_validation=metrics.f1_score(y_test, svm_linear_predict_labels_validation, average='macro')  
svm_linear_recall_validation=metrics.recall_score(y_test, svm_linear_predict_labels_validation, average='macro')
target_names = ['class 0', 'class 1']
print(classification_report(y_test, svm_linear_predict_labels_validation, target_names=target_names))

###test
svm_linear_acc_test=accuracy_score(y_test, svm_linear_predict_labels )
svm_linear_f1_test=metrics.f1_score(y_test, svm_linear_predict_labels, average='macro')  
svm_linear_recall_test=metrics.recall_score(y_test, svm_linear_predict_labels, average='macro')
#confusion_matrix(y_test, svm_linear_predict_labels)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, svm_linear_predict_labels, target_names=target_names))
print(svm_linear_acc_validation)
print(svm_linear_acc_test)

#######################
########svm(rbf)
svm_rbf = SVC(kernel='rbf',C=C_rbf,gamma= gamma)
svm_rbf.fit(n_x_train,y_train)
svm_rbf_predict_labels = svm_rbf.predict(n_x_test)# print accuracy of training data
svm_rbf_predict_labels_validation = svm_rbf.predict(n_x_validation)# print accuracy of training data

######svm metrics
########validation
svm_rbf_acc_validation=accuracy_score(y_test, svm_rbf_predict_labels_validation )
svm_rbf_f1_validation=metrics.f1_score(y_test, svm_rbf_predict_labels_validation, average='macro')
svm_rbf_recall_validation=metrics.recall_score(y_test, svm_rbf_predict_labels_validation, average='macro') 
target_names = ['class 0', 'class 1']
print(classification_report(y_test, svm_rbf_predict_labels_validation, target_names=target_names))
#########test
svm_rbf_acc_test=accuracy_score(y_test, svm_rbf_predict_labels )
svm_rbf_f1_test=metrics.f1_score(y_test, svm_rbf_predict_labels, average='macro')
svm_rbf_recall_test=metrics.recall_score(y_test, svm_rbf_predict_labels, average='macro') 
#confusion_matrix(y_test, svm_rbf_predict_labels)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, svm_rbf_predict_labels, target_names=target_names))
print(svm_rbf_acc_validation)
print(svm_rbf_acc_test)

#################
###random forest
rf = RandomForestClassifier(random_state = random_state,n_estimators= n_estimators,max_features=max_features,oob_score = oob_score)

rf.fit(n_x_train, y_train)
rf_predict_labels_test = rf.predict(n_x_test)
rf_predict_labels_validation = rf.predict(n_x_validation)

####rf metrics
#######validation
rf_acc_validation=accuracy_score(y_test, rf_predict_labels_validation )
rf_recall_validation=metrics.recall_score(y_test, rf_predict_labels_validation, average='macro')
rf_f1_validation=metrics.f1_score(y_test, rf_predict_labels_validation, average='macro')  
target_names = ['class 0', 'class 1']
print(classification_report(y_test, rf_predict_labels_validation, target_names=target_names))
#####test
rf_acc_test=accuracy_score(y_test, rf_predict_labels_test )
rf_recall_test=metrics.recall_score(y_test, rf_predict_labels_test, average='macro')
rf_f1_test=metrics.f1_score(y_test, rf_predict_labels_test, average='macro')  
target_names = ['class 0', 'class 1']
print(classification_report(y_test, rf_predict_labels_test, target_names=target_names))
print(rf_acc_validation)
print(rf_acc_test)


##########################
#####logistic regression
lr = LogisticRegression(solver = 'saga', penalty = 'l1')
lr.fit(n_x_train, y_train)
lr_predict_labels=lr.predict(n_x_test)
lr_predict_valid_labels = lr.predict(n_x_validation)
#######lr metrics

lr_acc_test=accuracy_score(y_test, lr_predict_labels )
lr_recall_test=metrics.recall_score(y_test, lr_predict_labels, average='macro')
lr_f1_test=metrics.f1_score(y_test, lr_predict_labels, average='macro')
lr_acc_valid = accuracy_score(y_validation, lr_predict_valid_labels)  
confusion_matrix(y_test, lr_predict_labels)

target_names = ['class 0', 'class 1']
print(lr_acc_test)
print(lr_acc_valid)
print(classification_report(y_test, lr_predict_labels, target_names=target_names))

###################################
######neural networks

from keras import models
from keras import layers
from keras import losses
from keras.layers import Dropout

size_hidden_layer = int(1.5 * len(x_use.columns))

#Model training
Model = models.Sequential()
Model.add(layers.Dense(len(x_use.columns), init = 'uniform',input_dim= len(x_use.columns)))
for i in range(3):
    Model.add(Dropout(0.5))
    Model.add(layers.Dense(size_hidden_layer, activation = 'relu'))
Model.add(layers.Dense(1,activation = "sigmoid"))
Model.summary()
Model.compile(optimizer = 'adam', loss = losses.binary_crossentropy,
              metrics = ['accuracy'])
history = Model.fit(n_x_train, y_train,epochs = 75, batch_size = 5,shuffle = False, 
                    validation_data = (n_x_validation,y_validation))
y_predict = Model.predict_classes(n_x_test)
y_predict_valid = Model.predict_classes(n_x_validation)

#change prediction format 
y_predict1 = []
mlp_predict = []
y_predict_valid1 = []
for i in range(0,len(y_predict)):
    y_predict1.append(y_predict[i][0])
    mlp_predict.append(str(y_predict[i][0]))
    y_predict_valid1.append(y_predict_valid[i][0])
    
y_validation1 = []
for i in range(455,len(y_test)+455):
    y_validation1.append(int(y_validation[i]))
    
y_test1 = []
for i in range(606,len(y_test)+606):
    y_test1.append(int(y_test[i]))


###get the metrics
from sklearn import metrics
print("accuray for test set: ", metrics.accuracy_score(y_test1,y_predict1))
print("accuracy for validation set", metrics.accuracy_score(y_validation1,y_predict_valid1))


mlp_acc_test = metrics.accuracy_score(y_test1,y_predict1)
mlp_recall_test = metrics.recall_score(y_test1,y_predict1, average='macro')
mlp_f1_test = metrics.f1_score(y_test1,y_predict1, average='macro')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

####ensemble voting###################################
print(svm_linear_predict_labels)
print(svm_rbf_predict_labels)
print(rf_predict_labels_test)
print(lr_predict_labels)
print(mlp_predict)

stack_voting = pd.DataFrame({'svm_linear': svm_linear_predict_labels,
                             'svm_rbf': svm_rbf_predict_labels,
                             'random_forest':rf_predict_labels_test,
                             'logistic_regression': lr_predict_labels,
                             'Neural_Network': mlp_predict})

stack_voting.svm_linear = pd.to_numeric(stack_voting.svm_linear, errors='coerce')
stack_voting.svm_rbf = pd.to_numeric(stack_voting.svm_rbf, errors='coerce')
stack_voting.random_forest = pd.to_numeric(stack_voting.random_forest, errors='coerce')
stack_voting.logistic_regression = pd.to_numeric(stack_voting.logistic_regression, errors='coerce')
stack_voting.Neural_Network = pd.to_numeric(stack_voting.Neural_Network, errors='coerce')

result_voting = stack_voting.sum(axis=1)

for i in range(len(result_voting)):
    if result_voting[i] >= 3:
        result_voting[i] = 1
    else:
        result_voting[i] = 0

stack_voting_acc = metrics.accuracy_score(y_test1,result_voting)
stack_voting_recall = metrics.recall_score(y_test1,result_voting, average='macro')
stack_voting_f1 = metrics.f1_score(y_test1,result_voting, average='macro')   

#####ensemble stacking###################################
stack_weight = {'svm_linear': svm_linear_acc_test,
                'svm_rbf': svm_rbf_acc_test,
                'random_forest':rf_acc_test,
                'logistic_regression': lr_acc_test,
                'Neural_Network': mlp_acc_test}

import operator
sorted_x = sorted(stack_weight.items(), key=operator.itemgetter(1))
weight_ratio = [0.1,0.1,0.2,0.2,0.4]

new_stack_weight = {}
for i in range(5):
    new_stack_weight.update({sorted_x[i][0]: weight_ratio[i]})

for key in new_stack_weight:
    stack_voting[key] = stack_voting[key] * new_stack_weight[key]

result_stacking = stack_voting.sum(axis=1)

for i in range(len(result_stacking)):
    if result_stacking[i] >= 0.5:
        result_stacking[i] = 1
    else:
        result_stacking[i] = 0

stacking_acc = metrics.accuracy_score(y_test1,result_stacking)
stacking_recall = metrics.recall_score(y_test1,result_stacking, average='macro')
stacking_f1 = metrics.f1_score(y_test1,result_stacking, average='macro')

#####store the results############################
if name == 'baseline':
    result_baseline.iloc[0,1]=rf_acc_test
    result_baseline.iloc[0,3]=rf_recall_test
    result_baseline.iloc[0,2]=rf_f1_test
    result_baseline.iloc[1,1]=mlp_acc_test
    result_baseline.iloc[1,2]=mlp_f1_test
    result_baseline.iloc[1,3]=mlp_recall_test
    result_baseline.iloc[2,1]=svm_rbf_acc_test
    result_baseline.iloc[2,2]=svm_rbf_f1_test
    result_baseline.iloc[2,3]=svm_rbf_recall_test
    result_baseline.iloc[3,1]=svm_linear_acc_test
    result_baseline.iloc[3,2]=svm_linear_f1_test
    result_baseline.iloc[3,3]=svm_linear_recall_test
    result_baseline.iloc[4,1]=lr_acc_test
    result_baseline.iloc[4,2]=lr_f1_test
    result_baseline.iloc[4,3]=lr_recall_test
    result_baseline.iloc[5,1]=stack_voting_acc
    result_baseline.iloc[5,2]=stack_voting_recall
    result_baseline.iloc[5,3]=stack_voting_f1
    result_baseline.iloc[6,1]=stacking_acc
    result_baseline.iloc[6,2]=stacking_recall
    result_baseline.iloc[6,3]=stacking_f1
    print(result_baseline)
    result_baseline.to_csv((name+'.csv'))
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    y = result_baseline.iloc[:,1]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(y):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, y, color = 'green')
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()
    
if name == 'mfcc':
    result_mfcc.iloc[0,1]=rf_acc_test
    result_mfcc.iloc[0,3]=rf_recall_test
    result_mfcc.iloc[0,2]=rf_f1_test
    result_mfcc.iloc[1,1]=mlp_acc_test
    result_mfcc.iloc[1,2]=mlp_f1_test
    result_mfcc.iloc[1,3]=mlp_recall_test
    result_mfcc.iloc[2,1]=svm_rbf_acc_test
    result_mfcc.iloc[2,2]=svm_rbf_f1_test
    result_mfcc.iloc[2,3]=svm_rbf_recall_test
    result_mfcc.iloc[3,1]=svm_linear_acc_test
    result_mfcc.iloc[3,2]=svm_linear_f1_test
    result_mfcc.iloc[3,3]=svm_linear_recall_test
    result_mfcc.iloc[4,1]=lr_acc_test
    result_mfcc.iloc[4,2]=lr_f1_test
    result_mfcc.iloc[4,3]=lr_recall_test
    result_mfcc.iloc[5,1]=stack_voting_acc
    result_mfcc.iloc[5,2]=stack_voting_recall
    result_mfcc.iloc[5,3]=stack_voting_f1
    result_mfcc.iloc[6,1]=stacking_acc
    result_mfcc.iloc[6,2]=stacking_recall
    result_mfcc.iloc[6,3]=stacking_f1
    print(result_mfcc)
    result_mfcc.to_csv((name+'.csv'))
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_mfcc.iloc[:,1]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, feature_quantity, color = 'green')
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()

if name == 'vocal_fold':
    result_vocal_fold.iloc[0,1]=rf_acc_test
    result_vocal_fold.iloc[0,3]=rf_recall_test
    result_vocal_fold.iloc[0,2]=rf_f1_test
    result_vocal_fold.iloc[1,1]=mlp_acc_test
    result_vocal_fold.iloc[1,2]=mlp_f1_test
    result_vocal_fold.iloc[1,3]=mlp_recall_test
    result_vocal_fold.iloc[2,1]=svm_rbf_acc_test
    result_vocal_fold.iloc[2,2]=svm_rbf_f1_test
    result_vocal_fold.iloc[2,3]=svm_rbf_recall_test
    result_vocal_fold.iloc[3,1]=svm_linear_acc_test
    result_vocal_fold.iloc[3,2]=svm_linear_f1_test
    result_vocal_fold.iloc[3,3]=svm_linear_recall_test
    result_vocal_fold.iloc[4,1]=lr_acc_test
    result_vocal_fold.iloc[4,2]=lr_f1_test
    result_vocal_fold.iloc[4,3]=lr_recall_test
    result_vocal_fold.iloc[5,1]=stack_voting_acc
    result_vocal_fold.iloc[5,2]=stack_voting_recall
    result_vocal_fold.iloc[5,3]=stack_voting_f1
    result_vocal_fold.iloc[6,1]=stacking_acc
    result_vocal_fold.iloc[6,2]=stacking_recall
    result_vocal_fold.iloc[6,3]=stacking_f1
    print(result_vocal_fold)
    result_vocal_fold.to_csv((name+'.csv'))
    
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_vocal_fold.iloc[:,1]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, feature_quantity, color = 'green')
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()

if name == 'wavelet':
    result_wavelet.iloc[0,1]=rf_acc_test
    result_wavelet.iloc[0,3]=rf_recall_test
    result_wavelet.iloc[0,2]=rf_f1_test
    result_wavelet.iloc[1,1]=mlp_acc_test
    result_wavelet.iloc[1,2]=mlp_f1_test
    result_wavelet.iloc[1,3]=mlp_recall_test
    result_wavelet.iloc[2,1]=svm_rbf_acc_test
    result_wavelet.iloc[2,2]=svm_rbf_f1_test
    result_wavelet.iloc[2,3]=svm_rbf_recall_test
    result_wavelet.iloc[3,1]=svm_linear_acc_test
    result_wavelet.iloc[3,2]=svm_linear_f1_test
    result_wavelet.iloc[3,3]=svm_linear_recall_test
    result_wavelet.iloc[4,1]=lr_acc_test
    result_wavelet.iloc[4,2]=lr_f1_test
    result_wavelet.iloc[4,3]=lr_recall_test
    result_wavelet.iloc[5,1]=stack_voting_acc
    result_wavelet.iloc[5,2]=stack_voting_recall
    result_wavelet.iloc[5,3]=stack_voting_f1
    result_wavelet.iloc[6,1]=stacking_acc
    result_wavelet.iloc[6,2]=stacking_recall
    result_wavelet.iloc[6,3]=stacking_f1
    print(result_wavelet)
    result_wavelet.to_csv((name+'.csv'))
    
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_wavelet.iloc[:,1]
    
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, feature_quantity, color = 'green')
    
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()

if name == 'time_frequency':
    result_time_frequency.iloc[0,1]=rf_acc_test
    result_time_frequency.iloc[0,3]=rf_recall_test
    result_time_frequency.iloc[0,2]=rf_f1_test
    result_time_frequency.iloc[1,1]=mlp_acc_test
    result_time_frequency.iloc[1,2]=mlp_f1_test
    result_time_frequency.iloc[1,3]=mlp_recall_test
    result_time_frequency.iloc[2,1]=svm_rbf_acc_test
    result_time_frequency.iloc[2,2]=svm_rbf_f1_test
    result_time_frequency.iloc[2,3]=svm_rbf_recall_test
    result_time_frequency.iloc[3,1]=svm_linear_acc_test
    result_time_frequency.iloc[3,2]=svm_linear_f1_test
    result_time_frequency.iloc[3,3]=svm_linear_recall_test
    result_time_frequency.iloc[4,1]=lr_acc_test
    result_time_frequency.iloc[4,2]=lr_f1_test
    result_time_frequency.iloc[4,3]=lr_recall_test
    result_time_frequency.iloc[5,1]=stack_voting_acc
    result_time_frequency.iloc[5,2]=stack_voting_recall
    result_time_frequency.iloc[5,3]=stack_voting_f1
    result_time_frequency.iloc[6,1]=stacking_acc
    result_time_frequency.iloc[6,2]=stacking_recall
    result_time_frequency.iloc[6,3]=stacking_f1
    print(result_time_frequency)
    result_time_frequency.to_csv((name+'.csv'))
    
    
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_time_frequency.iloc[:,1]
    
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, feature_quantity, color = 'green')
    
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()


if name == 'tqwt':
    result_tqwt.iloc[0,1]=rf_acc_test
    result_tqwt.iloc[0,3]=rf_recall_test
    result_tqwt.iloc[0,2]=rf_f1_test
    result_tqwt.iloc[1,1]=mlp_acc_test
    result_tqwt.iloc[1,2]=mlp_f1_test
    result_tqwt.iloc[1,3]=mlp_recall_test
    result_tqwt.iloc[2,1]=svm_rbf_acc_test
    result_tqwt.iloc[2,2]=svm_rbf_f1_test
    result_tqwt.iloc[2,3]=svm_rbf_recall_test
    result_tqwt.iloc[3,1]=svm_linear_acc_test
    result_tqwt.iloc[3,2]=svm_linear_f1_test
    result_tqwt.iloc[3,3]=svm_linear_recall_test
    result_tqwt.iloc[4,1]=lr_acc_test
    result_tqwt.iloc[4,2]=lr_f1_test
    result_tqwt.iloc[4,3]=lr_recall_test
    result_tqwt.iloc[5,1]=stack_voting_acc
    result_tqwt.iloc[5,2]=stack_voting_recall
    result_tqwt.iloc[5,3]=stack_voting_f1
    result_tqwt.iloc[6,1]=stacking_acc
    result_tqwt.iloc[6,2]=stacking_recall
    result_tqwt.iloc[6,3]=stacking_f1
    print(result_tqwt)
    result_tqwt.to_csv((name+'.csv'))
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    y = result_tqwt.iloc[:,1]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(algorithm_name)]
    for i, v in enumerate(y):
        plt.text(x_pos[i] - 0.25, v + 0.01, str(("%.3f" % round(v,3))))
    plt.bar(x_pos, y, color = 'green')
    plt.xlabel('Algorithm Used')
    plt.ylabel('Accuracy Rate')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, algorithm_name)
    plt.savefig((name + '.png'))
    plt.show()
