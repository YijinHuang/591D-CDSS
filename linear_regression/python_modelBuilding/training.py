#!/usr/bin/env python
import os, glob, sys
import numpy as np
import pandas as pd
import joblib

## Load all the data so we can quickly combine it and explore it. 
pfile = './CinC.pickle'
pfile_test = './CinC_test.pickle'
processed_pfile = 'processed_CINC.pickle'
processed_pfile_test = 'processed_CINC_test.pickle'
CINCdat = pd.read_pickle(pfile)
CINCdat_test = pd.read_pickle(pfile_test)
CINCdat_zScores = pd.read_pickle(processed_pfile)
CINCdat_test_zScores = pd.read_pickle(processed_pfile_test)

#### OPTION 1: LOGISTIC REGRESSION ####
## Build a logistic regression using all the training data
from sklearn.ensemble import RandomForestClassifier
# lreg = LogisticRegression(random_state=0, max_iter=1000)
randomForest = RandomForestClassifier(n_estimators = 100, criterion="log_loss", max_depth=8, max_features="log2", random_state=0)
data = CINCdat_zScores.loc[:, ~CINCdat_zScores.columns.isin(['SepsisLabel', 'patient']) ]
testData = CINCdat_test_zScores.loc[:, ~CINCdat_test_zScores.columns.isin(['SepsisLabel', 'patient']) ]
testLabels = CINCdat_test_zScores.SepsisLabel
dataLabels = CINCdat_zScores.SepsisLabel
randomForest.fit(data.values,dataLabels.values)
feature_imp = pd.Series(randomForest.feature_importances_, index = data.columns).sort_values(ascending = False)
print(feature_imp)
y_pred = randomForest.predict(testData)

# save model
joblib.dump(randomForest, 'randomForest.model')

from sklearn import metrics 
print()
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(testLabels, y_pred))

# print(randomForest.class_weight)
## Add the predictions
CINCdat_zScores = CINCdat_zScores.assign(probSepsisLR=randomForest.predict_proba(data)[::,1])
CINCdat_test_zScores = CINCdat_test_zScores.assign(probSepsisLR=randomForest.predict_proba(testData)[::,1])

## Quick but not necessarily great way to find a threshold. Also calculate the AUC
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisLR)
print('AUC:',round(roc_auc_score(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisLR),2)) # 0.72
print('AUC_test:',round(roc_auc_score(CINCdat_test_zScores.SepsisLabel,CINCdat_test_zScores.probSepsisLR),2)) # 0.7
thresh=round(thresholds[np.argmax(tpr - fpr)],4)
print('Threshold:',thresh)

# Quick calculation of utility score
print("assign train labels")
CINCdat = CINCdat.assign(SepsisLabelLR = (CINCdat_zScores.probSepsisLR>thresh).astype(int))
print("assign test labels")
CINCdat_test= CINCdat_test.assign(SepsisLabelLR = (CINCdat_test_zScores.probSepsisLR>thresh).astype(int))

print('evaluating')
import evaluate_sepsis_score as ev
# util = ev.evaluate_utility(CINCdat.patient,np.array(CINCdat_zScores.SepsisLabel),np.array(CINCdat.SepsisLabelLR))
# print(util) # 0.31570760013441407
util_test = ev.evaluate_utility(CINCdat_test.patient,np.array(CINCdat_test_zScores.SepsisLabel),np.array(CINCdat_test.SepsisLabelLR))
print(util_test) # 0.38775047041755845

#### OPTION 2: BOOSTED TREE #####
## Build a LightGBM model using all the training data
# import lightgbm as lgb
# train_data = lgb.Dataset(data=CINCdat_zScores.iloc[:,0:-3], label=CINCdat_zScores.SepsisLabel)
# param = {'objective': 'binary'}
# bst = lgb.train(param, train_data, 10)

## Add the predictions
# CINCdat_zScores = CINCdat_zScores.assign(probSepsisGBM=bst.predict(data=CINCdat_zScores.iloc[:,0:-3]))
# CINCdat_test_zScores = CINCdat_test_zScores.assign(probSepsisGBM=bst.predict(data=CINCdat_test_zScores.iloc[:,0:-3]))

## Quick but not necessarily great way to find a threshold. Also calculate the AUC
# from sklearn.metrics import roc_curve, roc_auc_score
# fpr, tpr, thresholds = roc_curve(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisGBM)
# print('AUC:',round(roc_auc_score(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisGBM),2)) # 0.83
# print('AUC_test:',round(roc_auc_score(CINCdat_test_zScores.SepsisLabel,CINCdat_test_zScores.probSepsisGBM),2)) # 0.72

# Save the model and get the threshold for use as a model
# bst.save_model('lightgbm.model')
# thresh=round(thresholds[np.argmax(tpr - fpr)],4)
# print('Threshold:',thresh)

# Quick calculation of utility score
# CINCdat = CINCdat.assign(SepsisLabelGBM = (CINCdat_zScores.probSepsisGBM>thresh).astype(int))
# CINCdat_test= CINCdat_test.assign(SepsisLabelGBM = (CINCdat_test_zScores.probSepsisGBM>thresh).astype(int))

# import evaluate_sepsis_score as ev
# util = ev.evaluate_utility(CINCdat.patient,np.array(CINCdat_zScores.SepsisLabel),np.array(CINCdat.SepsisLabelGBM))
# print(util) # 0.48660694527542603
# util_test = ev.evaluate_utility(CINCdat_test.patient,np.array(CINCdat_test_zScores.SepsisLabel),np.array(CINCdat_test.SepsisLabelGBM))
# print(util_test) # 0.3606544190522702
