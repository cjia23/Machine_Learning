#######this is for rfe
import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE #导入RFE库
from sklearn.linear_model import LogisticRegression #导入逻辑回归库
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv("./pd_speech_features.csv")
print(data.shape)
pd.isnull(data)
name = input("please enter the feature group name: (all_subset,out_mfcc,out_tqwt)")

########################
##### initialize 

if name == 'all_subset':
    result_all_subset_rfe=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logidtics regression','ensemble voting','ensemble stacking'],
                                    'accuracy':['1','2','3','4','5',1,0],
                                    'F1 score':['1','2','3','4','5',1,0],
                                    'recall score':['1','2','3','4','5',1,0]
                                })
elif name == 'out_mfcc':
    result_out_mfcc_rfe=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logidtics regression','ensemble voting','ensemble stacking'],
                                    'accuracy':['1','2','3','4','5',1,0],
                                    'F1 score':['1','2','3','4','5',1,0],
                                    'recall score':['1','2','3','4','5',1,0]
                                })
elif name == 'out_tqwt':
    result_out_tqwt_rfe=pd.DataFrame({'classification':['random forest','multilayer perceptron','SVM(RBF)','SVM(Linear)','Logidtics regression','ensemble voting','ensemble stacking'],
                                    'accuracy':['1','2','3','4','5',1,0],
                                    'F1 score':['1','2','3','4','5',1,0],
                                    'recall score':['1','2','3','4','5',1,0]
                                })

###### split features and labels
x=data.iloc[1:757,0:754]
y=data.iloc[1:757,754]
x.rename(columns={ x.columns[0]: "id" ,x.columns[1]: "gender"}, inplace=True)
########feature for rfe
x_out_mfcc=x.drop(x.columns[56:140],axis=1)
x_out_twqt=x.drop(x.columns[322:754],axis=1)
######select which feature groups you want to use (option includes x,x_out_mfcc,x_out_tqwt)

if name == 'all_subset':
    x_use = x
elif name == 'out_mfcc':
    x_use = x_out_mfcc
elif name == 'out_tqwt':
    x_use = x_out_twqt

#####split dataset
#####train: validation: test=6:2:2
x_train=x_use.iloc[0:454,:]
y_train=y[0:454]
x_validation=x_use.iloc[454:605,:]
y_validation=y[454:605]
x_test=x_use.iloc[605:757,:]
y_test=y[605:757]

######normalizaton
scaler= preprocessing.StandardScaler().fit(x_train)
n_x_train=scaler.transform(x_train)
n_x_validation=scaler.transform(x_validation)
n_x_test=scaler.transform(x_test)


######rfe
svc = SVC(kernel="linear")
#model = LogisticRegression() #设置算法为逻辑回归
rfe = RFE(svc, n_features_to_select=100) #选择100个最佳特征变量，并进行RFE
selector = rfe.fit(n_x_train, y_train) #进行RFE递归
selector.support_
selector.ranking_ 
new_x_train=n_x_train[:,selector.support_ ]
new_x_validation=n_x_validation[:,selector.support_]
new_x_test=n_x_test[:,selector.support_]
new_x_train.shape
new_x_validation.shape
new_x_test.shape

rfe_columns = selector.support_

if name == 'all_subset':
    rfe_baseline = np.array(rfe_columns[1:23])
    rfe_time_frequency = np.array(rfe_columns[23:34])
    rfe_vocal_fold = np.array(rfe_columns[34:56])
    rfe_mfcc = np.array(rfe_columns[56:56+84])
    rfe_wavelet = np.array(rfe_columns[140:140+182])
    rfe_tqwt = np.array(rfe_columns[322:754])
    
    n_baseline = len(rfe_baseline[rfe_baseline == True])
    n_time_frequency = len(rfe_time_frequency[rfe_time_frequency == True])
    n_vocal_fold = len(rfe_vocal_fold[rfe_vocal_fold == True])
    n_mfcc = len(rfe_mfcc[rfe_mfcc == True])
    n_wavelet = len(rfe_wavelet[rfe_wavelet == True])
    n_tqwt = len(rfe_tqwt[rfe_tqwt == True])
    
    feature_name = ['baseline', 'TF','vocal_fold', 'mfcc','wavelet','tqwt']
    feature_quantity = [n_baseline,n_time_frequency,n_vocal_fold,
                        n_mfcc,n_wavelet,n_tqwt]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(feature_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i]-0.1, v + 0.15, str(v))    
    plt.bar(x_pos, feature_quantity, color = 'green')
    
    plt.xlabel('Feature Group')
    plt.ylabel('Feature Quantity')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, feature_name)
    plt.savefig((name + '_rfe_feature.png'))
    plt.show()
    
if name == 'out_mfcc':
    rfe_baseline = np.array(rfe_columns[1:23])
    rfe_time_frequency = np.array(rfe_columns[23:34])
    rfe_vocal_fold = np.array(rfe_columns[34:56])
    rfe_mfcc = 0
    rfe_wavelet = np.array(rfe_columns[140-84:140+182-84])
    rfe_tqwt = np.array(rfe_columns[322-84:754-84])
    
    n_baseline = len(rfe_baseline[rfe_baseline == True])
    n_time_frequency = len(rfe_time_frequency[rfe_time_frequency == True])
    n_vocal_fold = len(rfe_vocal_fold[rfe_vocal_fold == True])
    n_mfcc = 0
    n_wavelet = len(rfe_wavelet[rfe_wavelet == True])
    n_tqwt = len(rfe_tqwt[rfe_tqwt == True])
    
    feature_name = ['baseline', 'TF','vocal_fold', 'mfcc','wavelet','tqwt']
    feature_quantity = [n_baseline,n_time_frequency,n_vocal_fold,
                        n_mfcc,n_wavelet,n_tqwt]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(feature_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i]-0.1, v + 0.15, str(v))    
    plt.bar(x_pos, feature_quantity, color = 'green')
    
    plt.xlabel('Feature Group')
    plt.ylabel('Feature Quantity')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, feature_name)
    plt.savefig((name + '_rfe_feature.png'))
    plt.show()    

if name == 'out_tqwt':
    rfe_baseline = np.array(rfe_columns[1:23])
    rfe_time_frequency = np.array(rfe_columns[23:34])
    rfe_vocal_fold = np.array(rfe_columns[34:56])
    rfe_mfcc = np.array(rfe_columns[56:56+84])
    rfe_wavelet = np.array(rfe_columns[140:140+182])
    rfe_tqwt = 0
    
    n_baseline = len(rfe_baseline[rfe_baseline == True])
    n_time_frequency = len(rfe_time_frequency[rfe_time_frequency == True])
    n_vocal_fold = len(rfe_vocal_fold[rfe_vocal_fold == True])
    n_mfcc = len(rfe_mfcc[rfe_mfcc == True])
    n_wavelet = len(rfe_wavelet[rfe_wavelet == True])
    n_tqwt = 0
    
    feature_name = ['baseline', 'TF','vocal_fold', 'mfcc','wavelet','tqwt']
    feature_quantity = [n_baseline,n_time_frequency,n_vocal_fold,
                        n_mfcc,n_wavelet,n_tqwt]
    plt.clf()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(feature_name)]
    for i, v in enumerate(feature_quantity):
        plt.text(x_pos[i]-0.1, v + 0.15, str(v))    
    plt.bar(x_pos, feature_quantity, color = 'green')
    
    plt.xlabel('Feature Group')
    plt.ylabel('Feature Quantity')
    plt.title(name + " Features Used")
    plt.xticks(x_pos, feature_name)
    plt.savefig((name + '_rfe_feature.png'))
    plt.show()

##############################################################
if name == 'all_subset':
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
elif name == 'out_mfcc':
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
elif name == 'out_tqwt':
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
    
#######classification
########svm(linear)
svm_linear = SVC(kernel='linear',C=C_linear)
svm_linear.fit(new_x_train,y_train)
svm_linear_predict_labels_validation = svm_linear.predict(new_x_validation)# print accuracy of training data
svm_linear_predict_labels = svm_linear.predict(new_x_test)# print accuracy of training data
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

##################
########svm(rbf)
svm_rbf = SVC(kernel='rbf',C=C_rbf,gamma=gamma)
svm_rbf.fit(new_x_train,y_train)
svm_rbf_predict_labels = svm_rbf.predict(new_x_test)# print accuracy of training data
svm_rbf_predict_labels_validation = svm_rbf.predict(new_x_validation)# print accuracy of training data

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
rf = RandomForestClassifier(random_state = random_state,n_estimators=n_estimators,max_features=max_features,oob_score = oob_score)
rf.fit(new_x_train, y_train)
rf_predict_labels_test = rf.predict(new_x_test)
rf_predict_labels_validation = rf.predict(new_x_validation)
####rf metrics
#######validation
rf_acc_validation=accuracy_score(y_test, rf_predict_labels_validation)
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
lr = LogisticRegression(solver = 'saga')
lr.fit(new_x_train, y_train)
lr_predict_labels=lr.predict(new_x_test)
lr_predict_valid_labels = lr.predict(new_x_validation)
#######lr metrics

lr_acc_test=accuracy_score(y_test, lr_predict_labels )
lr_recall_test=metrics.recall_score(y_test, lr_predict_labels, average='micro')
lr_f1_test=metrics.f1_score(y_test, lr_predict_labels, average='macro')
lr_acc_valid = accuracy_score(y_validation, lr_predict_valid_labels)  
confusion_matrix(y_test, lr_predict_labels)

target_names = ['class 0', 'class 1']
print(lr_acc_test)
print(lr_acc_valid)
print(classification_report(y_test, lr_predict_labels, target_names=target_names))


##############################################
######mlp

from keras import models
from keras import layers
from keras import losses
from keras.layers import Dropout

size_hidden_layer = 150

#Model training
Model = models.Sequential()
Model.add(layers.Dense(100, init = 'uniform',input_dim= 100))
for i in range(3):
    Model.add(Dropout(0.5))
    Model.add(layers.Dense(size_hidden_layer, activation = 'relu'))
Model.add(layers.Dense(1,activation = "sigmoid"))
Model.summary()
Model.compile(optimizer = 'adam', loss = losses.binary_crossentropy,
              metrics = ['accuracy'])
history = Model.fit(new_x_train, y_train,epochs = 75, batch_size = 5,shuffle = False,
                    validation_data = (new_x_validation, y_validation))
y_predict = Model.predict_classes(new_x_test)
y_predict_valid = Model.predict_classes(new_x_validation)

#change prediction format 
y_predict1 = []
y_predict_valid1 = []
for i in range(0,len(y_predict)):
    y_predict1.append(y_predict[i][0])
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
mlp_f1_test = metrics.f1_score(y_test1,y_predict1,average = 'macro')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#########################

####ensemble voting###################################
stack_voting = pd.DataFrame({'svm_linear': svm_linear_predict_labels,
                             'svm_rbf': svm_rbf_predict_labels,
                             'random_forest':rf_predict_labels_test,
                             'logistic_regression': lr_predict_labels,
                             'Neural_Network': y_predict1})

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




######all_subset_rfe
if name == 'all_subset':
    result_all_subset_rfe.iloc[0,1]=rf_acc_test
    result_all_subset_rfe.iloc[0,3]=rf_recall_test
    result_all_subset_rfe.iloc[0,2]=rf_f1_test
    result_all_subset_rfe.iloc[1,1]=mlp_acc_test
    result_all_subset_rfe.iloc[1,2]=mlp_f1_test
    result_all_subset_rfe.iloc[1,3]=mlp_recall_test
    result_all_subset_rfe.iloc[2,1]=svm_rbf_acc_test
    result_all_subset_rfe.iloc[2,2]=svm_rbf_f1_test
    result_all_subset_rfe.iloc[2,3]=svm_rbf_recall_test
    result_all_subset_rfe.iloc[3,1]=svm_linear_acc_test
    result_all_subset_rfe.iloc[3,2]=svm_linear_f1_test
    result_all_subset_rfe.iloc[3,3]=svm_linear_recall_test
    result_all_subset_rfe.iloc[4,1]=lr_acc_test
    result_all_subset_rfe.iloc[4,2]=lr_f1_test
    result_all_subset_rfe.iloc[4,3]=lr_recall_test
    result_all_subset_rfe.iloc[5,1]=stack_voting_acc
    result_all_subset_rfe.iloc[5,2]=stack_voting_f1
    result_all_subset_rfe.iloc[5,3]=stack_voting_recall
    result_all_subset_rfe.iloc[6,1]=stacking_acc
    result_all_subset_rfe.iloc[6,2]=stacking_f1
    result_all_subset_rfe.iloc[6,3]=stacking_recall
    print(result_all_subset_rfe)
    result_all_subset_rfe.to_csv((name+'.csv'))
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_all_subset_rfe.iloc[:,1]
    
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
######_out_mfcc_rfe
if name == 'out_mfcc':
    result_out_mfcc_rfe.iloc[0,1]=rf_acc_test
    result_out_mfcc_rfe.iloc[0,3]=rf_recall_test
    result_out_mfcc_rfe.iloc[0,2]=rf_f1_test
    result_out_mfcc_rfe.iloc[1,1]=mlp_acc_test
    result_out_mfcc_rfe.iloc[1,2]=mlp_f1_test
    result_out_mfcc_rfe.iloc[1,3]=mlp_recall_test
    result_out_mfcc_rfe.iloc[2,1]=svm_rbf_acc_test
    result_out_mfcc_rfe.iloc[2,2]=svm_rbf_f1_test
    result_out_mfcc_rfe.iloc[2,3]=svm_rbf_recall_test
    result_out_mfcc_rfe.iloc[3,1]=svm_linear_acc_test
    result_out_mfcc_rfe.iloc[3,2]=svm_linear_f1_test
    result_out_mfcc_rfe.iloc[3,3]=svm_linear_recall_test
    result_out_mfcc_rfe.iloc[4,1]=lr_acc_test
    result_out_mfcc_rfe.iloc[4,2]=lr_f1_test
    result_out_mfcc_rfe.iloc[4,3]=lr_recall_test
    result_out_mfcc_rfe.iloc[5,1]=stack_voting_acc
    result_out_mfcc_rfe.iloc[5,2]=stack_voting_f1
    result_out_mfcc_rfe.iloc[5,3]=stack_voting_recall
    result_out_mfcc_rfe.iloc[6,1]=stacking_acc
    result_out_mfcc_rfe.iloc[6,2]=stacking_f1
    result_out_mfcc_rfe.iloc[6,3]=stacking_recall
    print(result_out_mfcc_rfe)
    result_out_mfcc_rfe.to_csv((name+'.csv'))
    
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_out_mfcc_rfe.iloc[:,1]
    
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
    
######_out_tqwt_rfe
if name == 'out_tqwt':
    result_out_tqwt_rfe.iloc[0,1]=rf_acc_test
    result_out_tqwt_rfe.iloc[0,3]=rf_recall_test
    result_out_tqwt_rfe.iloc[0,2]=rf_f1_test
    result_out_tqwt_rfe.iloc[1,1]=mlp_acc_test
    result_out_tqwt_rfe.iloc[1,2]=mlp_f1_test
    result_out_tqwt_rfe.iloc[1,3]=mlp_recall_test
    result_out_tqwt_rfe.iloc[2,1]=svm_rbf_acc_test
    result_out_tqwt_rfe.iloc[2,2]=svm_rbf_f1_test
    result_out_tqwt_rfe.iloc[2,3]=svm_rbf_recall_test
    result_out_tqwt_rfe.iloc[3,1]=svm_linear_acc_test
    result_out_tqwt_rfe.iloc[3,2]=svm_linear_f1_test
    result_out_tqwt_rfe.iloc[3,3]=svm_linear_recall_test
    result_out_tqwt_rfe.iloc[4,1]=lr_acc_test
    result_out_tqwt_rfe.iloc[4,2]=lr_f1_test
    result_out_tqwt_rfe.iloc[4,3]=lr_recall_test
    result_out_tqwt_rfe.iloc[5,1]=stack_voting_acc
    result_out_tqwt_rfe.iloc[5,2]=stack_voting_f1
    result_out_tqwt_rfe.iloc[5,3]=stack_voting_recall
    result_out_tqwt_rfe.iloc[6,1]=stacking_acc
    result_out_tqwt_rfe.iloc[6,2]=stacking_f1
    result_out_tqwt_rfe.iloc[6,3]=stacking_recall
    print(result_out_tqwt_rfe)
    result_out_tqwt_rfe.to_csv((name+'.csv'))
    
    algorithm_name = ['RF', 'MLP',
                    'SVM_rbf', 'SVM_linear'
                    ,'LR','Voting','Stacking']
    feature_quantity = result_out_tqwt_rfe.iloc[:,1]
    
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



