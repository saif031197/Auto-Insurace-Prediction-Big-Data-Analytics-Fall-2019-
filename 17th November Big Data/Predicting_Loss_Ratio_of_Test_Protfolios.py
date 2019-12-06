# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:04:01 2019

@author: sharm
"""

import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge, RidgeCV, LassoCV

full_data = pd.read_csv(r"C:\Users\sharm\Desktop\Books\Semester 3\Big Data For Competative Advances\Big Data Project\Feature_Engineered_Dataset_Nov17.csv")
full_data = full_data.drop(["Unnamed: 0"], axis=1)

test_data = pd.read_csv(r"C:\Users\sharm\Desktop\Books\Semester 3\Big Data For Competative Advances\Big Data Project\Testing Portfolio.csv")

# Shuffle the data before spliting
full_data=full_data.sample(frac=1, random_state=10).reset_index(drop=True)

train = full_data[:]
y_train = pd.DataFrame(train["Loss_Amount"])
train = train.drop(["Loss_Amount"], axis=1)

test = test_data.drop(["Portfolio"], axis=1)
test= test.drop(["Unnamed: 0"], axis=1)

print("Training Dataset = ", train.shape)
print("Testing Dataset = ",test.shape)
print("Train Predict = ",y_train.shape)


# Scaling the training Dataset
# We have train, y_train and test
scaler = preprocessing.StandardScaler().fit(train)
train = scaler.transform(train)

# scale test dataset
test = scaler.transform(test)

print('Means = ', scaler.mean_)
print('Standard Deviations = ', scaler.scale_)


## Modeling and Predictions

# L1_cv
l1_cv = LassoCV(cv=5, max_iter=10000)
#100 Regularization coefficients evenly spaced between 0.1 and 1000
l1_cv.alphas = tuple(np.linspace(0.1,1000,100))
l1_cv.fit(train, y_train)
res_l1 = l1_cv.predict(test)
print(res_l1)

# L2_cv
l2_cv = RidgeCV(cv=None, store_cv_values=True)
#100 Regularization coefficients evenly spaced between 0.1 and 1000
l2_cv.alphas = tuple(np.linspace(0.1,10000,100))
l2_cv.fit(train, y_train)
res_l2 = l2_cv.predict(test)

print(res_l2)

# res_xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)
model_xgb.fit(train,y_train)
res_xgb = model_xgb.predict(test)
print(res_xgb)

# res_lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(train,y_train)
res_lgb = model_lgb.predict(test)
print(res_lgb)

# Gboost
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(train,y_train)
res_gboost = GBoost.predict(test)
print(res_gboost)

# Kernel Ridge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR.fit(train,y_train)
res_KRR = KRR.predict(test)
print(res_KRR)


# ENet
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(train,y_train)
res_ENet = ENet.predict(test)
print(res_ENet)

import itertools

# Average
res_ENet = res_ENet.tolist()
res_KRR = res_KRR.tolist()
res_gboost = res_gboost.tolist()
res_lgb = res_lgb.tolist()
res_xgb = res_xgb.tolist()
res_l2 = res_l2.tolist()
res_l1 = res_l1.tolist()

res_KRR = list(itertools.chain(*res_KRR))
print(res_KRR)

res_l2 = list(itertools.chain(*res_l2))
print(len(res_l2))

Sum = []
for i in range(len(res_ENet)):
    Sum.append(res_ENet[i]+res_KRR[i]+res_gboost[i]+res_lgb[i]+res_xgb[i]+res_l2[i]+res_l1[i])

print(len(Sum))

Sum = np.array(Sum)
Sum = Sum/7
print(len(Sum))


test_premium = test_data["Annual_Premium"].values
test_premium = test_premium.tolist()
test_premium = np.array(test_premium)

log_ratio = math.log(Sum/test_premium)
print(len(log_ratio))


print(Sum)

Result = pd.DataFrame()
Result["ID"] = test_data["Portfolio"]


log_ratio=log_ratio.tolist()

log_ln = []
for i in range(len(log_ratio)):
    if log_ratio[i] > 0:   
      print(log_ratio[i])
      log_ln.append(math.log(float(log_ratio[i])))
    else:
        print(log_ratio[i])
        log_ln.append(0)
   
print(log_ln)
    
Result["ln_LR"] = log_ln 

print(Result)

type(Result)
Result.to_csv("Result.csv")











