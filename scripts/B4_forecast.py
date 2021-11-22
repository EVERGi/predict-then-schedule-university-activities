#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, inspect
import sys
import fnmatch
os.chdir(os.path.dirname(os.getcwd()))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt


from prophet import Prophet
import lightgbm as lgb
import optuna

from sklearn import metrics
#from src.utils.functions import validation, calculate_mase


# In[2]:


from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.feature_selection import VarianceThreshold


# In[3]:


plt.style.use('fivethirtyeight')

def objective(trial, input_X, input_y):
    param = {
        'metric': 'auc',
        'class_weight': 'balanced',
        'method': trial.suggest_categorical("method", ['gbdt', 'dart', 'goss']),
        'random_state': 48,
        "num_iterators": trial.suggest_categorical("num_iterators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15, 20, 25]), 
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
#         'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    train_date = '01-Sep-2020'
    split_date = '01-Oct-2020'
    correlated_features = set()
    correlation_matrix = input_X.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > trial.suggest_categorical("correlation_value", [0.6, 0.7, 0.8, 0.9, 1]):
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    
    input_X = input_X[(input_y - input_y.mean())/input_y.std(ddof=0) < 3]
    input_y = input_y[(input_y - input_y.mean())/input_y.std(ddof=0) < 3]
    
    # Make temporary division
    X_train, y_train = input_X.loc[input_X.index < train_date].copy(), input_y.loc[input_y.index < train_date].copy()
    X_eval, y_eval = input_X.loc[(input_X.index >= train_date) & (input_X.index < split_date)].copy(), input_y.loc[(input_y.index >= train_date) & (input_y.index < split_date)].copy()
    
    n = len(X_train)
    
    X_train, y_train = X_train.fillna(value=0), y_train.fillna(value=0)
    X_eval, y_eval = X_eval.fillna(value=0), y_eval.fillna(value=0)
                
    X_train.drop(labels=correlated_features, axis=1, inplace=True)
    X_eval.drop(labels=correlated_features, axis=1, inplace=True)

    model = lgb.LGBMClassifier(**param)
    
    model.fit(
        X_train, 
        y_train,                     
        eval_set=[(X_eval, y_eval)],
        eval_metric='auc',
        early_stopping_rounds=50,
        callbacks=[
            LightGBMPruningCallback(trial, "auc")
        ],  # Add a pruning callback
    )
    
    preds = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, preds)
    trial.set_user_attr(key="best_booster", value=model)
    return mae

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[4]:


import_dir = './data/lgbm_inputs_phase2/'
plots_dir = './plots/'
log_dir = './logs/'
model_dir = './models/'
namerun = 'classifier_300'
filelist = os.listdir(import_dir)
# buildings = fnmatch.filter(list(filelist),'Build*')
# print(buildings)
mase = {}
smape = {}
predictions = {}
results_df = pd.DataFrame()
i = 0
filename = 'Building4.csv'
path = os.path.join(import_dir, filename)
input_ds = pd.read_csv(path, index_col='datetime', parse_dates=[0])
start_date = '01-Apr-2020'
input_ds.index = input_ds.index.sort_values()
# THE DATA AFTER WHICH THE TEST SERIES STARTS
train_date = '01-Oct-2020'
split_date = '01-Nov-2020'

# Divide features and labels 
input_X, input_y = input_ds.iloc[:,1:], input_ds.iloc[:,0]


# Create study that minimizes
study = optuna.create_study(direction="maximize")

# Pass additional arguments inside another function
func = lambda trial: objective(trial, input_X, input_y)

# Start optimizing with 100 trials
study.optimize(func, n_trials=300, callbacks=[callback])

correlated_features = set()
correlation_matrix = input_X.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > study.best_trial.params['correlation_value']:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

X_train, y_train = input_X.loc[input_X.index < train_date].copy(), input_y.loc[input_y.index < train_date].copy()
X_test, y_test = input_X.loc[input_X.index >= split_date].copy(), input_y.loc[input_y.index >= split_date].copy()

n = len(X_train)

X_test = X_test.fillna(value=0)
X_test.drop(labels=correlated_features, axis=1, inplace=True)

print(f"Optimized MAE: {study.best_value:.5f}")
seas = 28 * 24 * 4
h = 2976
y = input_ds[:'Sep-2020']['value']
persistence = np.sum(np.abs(y[seas:n].reset_index(drop=True) - y[:n-seas].reset_index(drop=True)), axis=0)
print(filename)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
f = open(log_dir+namerun+"{}_log.txt".format(os.path.splitext(filename)[0]), "w+")
f.write(filename)
f.write('\n')
f.write('MAE error is:{}\n'.format(study.best_value))
f.write('Number of finished trials:{}\n'.format(len(study.trials)))
f.write('Best trial:{}\n'.format(study.best_trial.params))
f.close()
best_model=study.user_attrs["best_booster"]
pickle.dump(best_model, open(model_dir+namerun+"_best_{}.pkl".format(os.path.splitext(filename)[0]), "wb"))

lgbmodel = pickle.load(open(model_dir+namerun+"_best_{}.pkl".format(os.path.splitext(filename)[0]), "rb"))

# predict the test dataset
forecast = pd.DataFrame(lgbmodel.predict(X_test), index=X_test.index)

plotImp(lgbmodel, X_test, num = 20, fig_size = (40, 20))
# FILL ZEROS FOR PLOTTING
input_ds = input_ds.fillna(value=0)
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
input_ds.loc['Sep-2020':'Oct-2020']['value'].plot(ax=ax, style=['-'])
forecast.plot(ax=ax, style=['-'])
ax.set_xbound(lower='2020-09', upper='2020-12')
plot = plt.suptitle(filename, y=1.01)
plot = plt.title('September-Octiber Actual and November Forecasted')
plt.savefig(plots_dir+namerun+'_{}.png'.format(os.path.splitext(filename)[0]))
s = 28 * 24 * 4
h = 2976
y = input_ds['value']
predictions[filename] = list(forecast)
results_df = pd.concat([results_df, forecast.rename({0:os.path.splitext(filename)[0]}, axis=1).T])


# In[7]:


results_df
results_df.to_csv('./results/buildings_B4.csv', header=False)


# In[ ]:




