#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, inspect
import sys
#os.chdir(os.path.dirname(os.getcwd()))


# In[3]:


import fnmatch

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback


from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from scipy import stats


# In[5]:


import_dir = './data/lgbm_inputs_phase2/'
# THIS IS THE LIST OF TIMESERIES WHERE NANS ARE FILLED WITH 0 FOR TRAINING
filelist = os.listdir(import_dir)
if ('.ipynb_checkpoints' in filelist):
    filelist.remove('.ipynb_checkpoints')
#buildings = fnmatch.filter(list(filelist),'Building*')
#buildings.remove('Building4.csv')
buildings = ['Building0.csv', 'Building1.csv', 'Building3.csv']
print(buildings)
plots_dir = './plots/'
log_dir = './logs/'
model_dir = './models/'
namerun = 'november_xp'
prophets_dir = './data/prophets/'
feature_plots_dir = './plots/importances/'
drop_features = ['y', 'value']
predictions = {}
results_df = pd.DataFrame()
prophets = {}
transform = 'log'


# In[7]:


for i,filename in enumerate(buildings):
    path = os.path.join(import_dir, filename)
    input_ds = pd.read_csv(path, index_col='datetime', parse_dates=[0])
    input_ds = input_ds.rename({'value':'y'},axis=1)
    input_ds['ds'] = input_ds.index
    input_ds.index = input_ds.index.sort_values()
    # THE DATA AFTER WHICH THE TEST SERIES STARTS
    split_date = '01-Nov-2020'
    input_ds['value'] = input_ds['y']
    input_ds['y'] = np.log1p(input_ds['y'])
    # Separate dataframe for application in Prophet
    input_ds_prop = input_ds.dropna().copy()

    input_ds_train = input_ds_prop.loc[input_ds_prop.index < split_date].copy()
    input_ds_train = input_ds_train[(input_ds_train.value - input_ds_train.value.mean())/input_ds_train.value.std(ddof=0) < 3]
    param_grid = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 0.1,
        'changepoint_range': 0.9,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
    }

    prop = Prophet(**param_grid)
    prop.add_country_holidays(country_name='Australia')
    #prop.add_regressor('max_temp_moorabbin')
    prop.fit(input_ds_train)

    future = prop.make_future_dataframe(periods=2976, freq='15T')
    #future = future.join(input_ds['max_temp_moorabbin'], how='outer', on='ds')

    forecast = prop.predict(future)
    fig1 = prop.plot(forecast)
    fig2 = prop.plot_components(forecast)
    prophets[filename] = forecast
    df_name = filename + '_' + transform
    forecast.to_csv(os.path.join(prophets_dir, df_name), index=False)


# In[9]:


plt.style.use('fivethirtyeight')

def plot_predictions(input_ds_all, name):
    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = input_ds_all[['value','Prediction']].plot(ax=ax,
                                                  style=['-','-'])
    ax.set_xbound(lower='2020-09', upper='2020-10')
    plot = plt.suptitle(name, y=1.01)
    plot = plt.title('September 2020 Forecast vs Actuals')
    plt.show()

def plotImp(model, X , num = 20, fig_size = (40, 20)):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    #sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    #plt.savefig('lgbm_importances-01.png')
    plt.show()

def weekly_persistence(input_ds_train):
    predictions = pd.Series(dtype='float64')
    #input_ds_train.value[-672:].plot()
    input_ds_train = input_ds_train.resample('15T').mean()
    input_ds_train = input_ds_train.fillna(value=0)
    last_week = input_ds_train.value[-672:]
    for i in range(len(X_test)//672 + 1):
    # get the data for the prior week
        last_week.index = last_week.index.shift(freq='7D')
        #last_week.plot()
        predictions = predictions.append(last_week)
        #print(predictions)
    #print(predictions)
    predictions = predictions['2020-10']
    return predictions

def objective(trial, X, y, X_valid, y_valid):
    param = {
        'metric': 'l1',
        'random_state': 48,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15, 20, 25]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
#         'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
#         'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }

    #model = lgb.LGBMRegressor(**param)

    ds_train = lgb.Dataset(X, y, categorical_feature=['working', 'weekday'])
    ds_validate = lgb.Dataset(X_valid, y_valid, categorical_feature=['working', 'weekday'])

    model= lgb.train(
        train_set = ds_train,
        #eval_set=[ds_validate],
        valid_sets=[ds_validate],
        #eval_metric='l1',
        early_stopping_rounds=50,
        params=param,
        callbacks=[
            LightGBMPruningCallback(trial, "l1")
        ],  # Add a pruning callback
    )

    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    trial.set_user_attr(key="best_booster", value=model)
    return mae

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[10]:


for i,filename in enumerate(buildings):
    path = os.path.join(import_dir, filename)
    input_ds = pd.read_csv(path, index_col='datetime', parse_dates=[0])
    input_ds = input_ds.rename({'value':'y'},axis=1)
    input_ds['value'] = input_ds['y']
    input_ds['y'] = np.log1p(input_ds['y'])
    input_ds.index = input_ds.index.sort_values()
    #forecast = prophets[filename]
    df_name = filename + '_' + transform
    forecast = pd.read_csv(os.path.join(prophets_dir, df_name), index_col='ds', parse_dates=[0])

    input_ds = input_ds.join(forecast.drop(['yhat', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper', 'daily_lower', 'daily_upper', 'weekly_lower', 'weekly_upper', 'yhat_upper', 'yhat_lower', 'ds', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper', 'Anzac Day', 'Anzac Day_lower', 'Anzac Day_upper', 'Australia Day',
       'Australia Day_lower', 'Australia Day_upper',
       'Australia Day (Observed)', 'Australia Day (Observed)_lower',
       'Australia Day (Observed)_upper', 'Boxing Day', 'Boxing Day_lower',
       'Boxing Day_upper', 'Boxing Day (Observed)',
       'Boxing Day (Observed)_lower', 'Boxing Day (Observed)_upper',
       'Christmas Day', 'Christmas Day_lower', 'Christmas Day_upper',
       'Christmas Day (Observed)', 'Christmas Day (Observed)_lower',
       'Christmas Day (Observed)_upper', 'Easter Monday',
       'Easter Monday_lower', 'Easter Monday_upper', 'Good Friday',
       'Good Friday_lower', 'Good Friday_upper', 'New Year\'s Day',
       'New Year\'s Day_lower', 'New Year\'s Day_upper',
       'New Year\'s Day (Observed)', 'New Year\'s Day (Observed)_lower',
       'New Year\'s Day (Observed)_upper'], axis=1, errors='ignore'))

    if  (filename != 'Building5.csv'):
        input_ds = input_ds.join(input_ds['y'].shift(freq='35D'),how='inner',rsuffix='-3360')
#     if 'yearly' in input_ds.columns:
#         input_ds['additive_term'] = input_ds['y'] - input_ds['trend'] - input_ds['yearly'] - input_ds['holidays']
#     else:
#         input_ds['additive_term'] = input_ds['y'] - input_ds['trend'] - input_ds['holidays']

    if (filename == 'Building5.csv'):
        train_date = '01-Aug-2020'
        val_date = '01-Sep-2020'
        split_date = '01-Nov-2020'
        input_ds_train = input_ds.loc[input_ds.index < train_date].copy()
        input_ds_eval = input_ds.loc[(input_ds.index >= train_date) & (input_ds.index < val_date)].copy()
        input_ds_test = input_ds.loc[input_ds.index >= split_date].copy()
    else:
        train_date = '01-Oct-2020'
        split_date = '01-Nov-2020'
        input_ds_train = input_ds.loc[input_ds.index < train_date].copy()
        input_ds_eval = input_ds.loc[(input_ds.index >= train_date) & (input_ds.index < split_date)].copy()
        input_ds_test = input_ds.loc[input_ds.index >= split_date].copy()

    n = len(input_ds_train)

    if (filename in ['Building5.csv', 'Building4.csv']):
        input_ds_train = input_ds_train.fillna(value=0)
    else:
        input_ds_train = input_ds_train.dropna()
    input_ds_eval = input_ds_eval.dropna()
    input_ds_test = input_ds_test.fillna(value=0)

    X_train, y_train = input_ds_train.drop(drop_features, axis=1), input_ds_train['y']
    X_eval, y_eval = input_ds_eval.drop(drop_features, axis=1), input_ds_eval['y']
    X_test, y_test = input_ds_test.drop(drop_features, axis=1), input_ds_test['y']
    X_test = X_test.fillna(value=0)

    # Create study that minimizes
    study = optuna.create_study(direction="minimize")

    # Pass additional arguments inside another function
    func = lambda trial: objective(trial, X_train, y_train, X_valid=X_eval, y_valid=y_eval)

    # Start optimizing with 100 trials
    study.optimize(func, n_trials=200, callbacks=[callback])

    print(f"Optimized MAE: {study.best_value:.5f}")
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
    # save the model
    pickle.dump(best_model, open(model_dir+namerun+"_best_{}.pkl".format(os.path.splitext(filename)[0]), "wb"))
    # load the model
    lgbmodel = pickle.load(open(model_dir+namerun+"_best_{}.pkl".format(os.path.splitext(filename)[0]), "rb"))
    # predict the test dataset
    add_forecast = pd.DataFrame(lgbmodel.predict(X_test), index=input_ds_test.index, columns=['additive'])
#     if 'yearly' in input_ds_test.columns:
#         forecast = add_forecast['additive'] + input_ds_test['trend'] + input_ds_test['yearly'] + input_ds_test['holidays']
#     else:
#         forecast = add_forecast['additive'] + input_ds_test['trend'] + input_ds_test['holidays']
    forecast = pd.DataFrame(add_forecast['additive'], index=input_ds_test.index, columns=['additive'])
    forecast = np.expm1(forecast)
    if (filename == 'Building4.csv'):
        forecast = round(forecast)

    #Plotting feature importance
    f_imp, ax_imp = plt.subplots(1, figsize=(30,15))
    plt.gcf().subplots_adjust(left=0.5)
    lgb.plot_importance(lgbmodel, height=0.3, ax=ax_imp)
    plt.savefig(feature_plots_dir+namerun+'_{}.png'.format(os.path.splitext(filename)[0]))

    #plotImp(lgbmodel, X_test, num = 20, fig_size = (40, 20))
    # FILL ZEROS FOR PLOTTING
    input_ds = input_ds.fillna(value=0)
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    input_ds.loc['Sep-2020':'Oct-2020']['value'].plot(ax=ax, style=['-'])
    forecast.plot(ax=ax, style=['-'])
    ax.set_xbound(lower='2020-09', upper='2020-12')
    plot = plt.suptitle(filename, y=1.01)
    plot = plt.title('September Actual and October Forecasted')
    plt.savefig(plots_dir+namerun+'_{}.png'.format(os.path.splitext(filename)[0]))
    predictions[filename] = list(forecast)
    results_df = pd.concat([results_df, forecast.rename({'additive':os.path.splitext(filename)[0]}, axis=1).T])


# In[13]:


results_df.index = [os.path.splitext(filename)[0] for filename in buildings]
results_df.to_csv('./results/buildings_B013.csv', header=False)
results_df


# In[ ]:
