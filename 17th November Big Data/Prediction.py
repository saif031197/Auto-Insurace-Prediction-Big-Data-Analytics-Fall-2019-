# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:25:48 2019

@author: saifi
"""

import pandas as pd
import numpy as np
import math
df=pd.read_csv('Clean_Dataset_17nov.csv')

def loss_or_not(x):
    if(x>1.0):
        return 1
    else:
        return 0
    

df['Loss_or_Not']=df['Loss_Amount'].apply(loss_or_not)

x=df.values[:,:30]
y=df.values[:,31]

from sklearn.ensemble import BaggingClassifier
from sklearn import tree

model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(x, y)



df2=df.loc[df['Loss_or_Not']==1]

import xgboost as xgb

x2=df2.values[:,:30]
y2=df2.values[:,30]
data_dmatrix = xgb.DMatrix(data=x2,label=y2)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', booster="gbtree", nthread=-1, colsample_bytree = 0.7, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(x2,y2)

final_portfolio=pd.DataFrame(columns=['ID','ln_LR'])
error=[]
for x in range(1,601):
    try:
        df_test=pd.read_csv('test_portfolio_cleaned17nov_'+str(x)+'.csv')
        #df_test=df_test.drop(['Unnamed: 0'], axis=1)
        x_test=df_test.values[:,:30]
        preds=model.predict(x_test)
        df_test['Loss_or_Not']=preds
        df_test_reg=df_test.loc[df_test['Loss_or_Not']==1]
        x_test_reg=df_test_reg.values[:,:30]
        preds_reg = xg_reg.predict(x_test_reg)
        df_test_reg['Loss']=preds_reg
        df_test=df_test.loc[df_test['Loss_or_Not']==0]
        df_test['Loss']=0
        df_test=df_test.append(df_test_reg)
        #df_test['Annual_Premium'].sum()
        #df_test['Loss'].sum()
        
        loss_ratio=df_test['Loss'].sum()/df_test['Annual_Premium'].sum()
        #math.log(loss_ratio)

        final_portfolio=final_portfolio.append({'ID':'portfolio_'+str(x), 'ln_LR':math.log(loss_ratio)},ignore_index=True)
    except:
        error.append(x)
        pass
    
final_portfolio.to_csv('result.csv',index=False)