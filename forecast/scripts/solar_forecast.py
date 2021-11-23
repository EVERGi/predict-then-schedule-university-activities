#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# os.chdir(os.path.dirname(os.getcwd()))


# In[3]:


import fnmatch
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt


import lightgbm as lgb
import optuna

from sklearn import metrics

from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.feature_selection import VarianceThreshold


# In[31]:


import_dir = './data/lgbm_inputs_phase2/'
plots_dir = './plots/'
log_dir = './logs/'
model_dir = './models/'
namerun = 'november_solar'
feature_plots_dir = './plots/importances/'
filelist = os.listdir(import_dir)
if ('.ipynb_checkpoints' in filelist):
    filelist.remove('.ipynb_checkpoints')
solar = fnmatch.filter(list(filelist),'Solar*')
print(solar)
mase = {}
smape = {}
predictions = {}
results_df = pd.DataFrame()


# In[32]:


plt.style.use('fivethirtyeight')

def plot_predictions(input_ds_all, name):
    # Plot the forecast with the actuals
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = input_ds_all[['value','Prediction']].plot(ax=ax,
                                                  style=['-','-'])
    ax.set_xbound(lower='2020-09', upper='2020-11')
    plot = plt.suptitle(name, y=1.01)
    plt.title('September 2020 Forecast vs Actuals')
    plt.show()

def plotImp(model, X , num = 20, fig_size = (40, 20)):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    #sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()


def objective(trial, input_X, input_y):

    param = {
        'metric': 'l1',
        'method': trial.suggest_categorical("method", ['gbdt', 'dart', 'goss']),
        'random_state': 48,
        "num_iterators": trial.suggest_categorical("num_iterators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 5, 10, 15, 20, 25]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    }

    correlated_features = set()
    correlation_matrix = input_X.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > trial.suggest_categorical("correlation_value", [0.6, 0.7, 0.8, 0.9, 1]):
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    train_date = '01-Oct-2020'
    split_date = '01-Nov-2020'
    # Make temporary division
    X_train, y_train = input_X.loc[input_X.index < train_date].copy(), input_y.loc[input_y.index < train_date].copy()
    X_eval, y_eval = input_X.loc[(input_X.index >= train_date) & (input_X.index < split_date)].copy(), input_y.loc[(input_y.index >= train_date) & (input_y.index < split_date)].copy()

    len(X_train)

    X_train, y_train = X_train.fillna(value=0), y_train.fillna(value=0)
    X_eval, y_eval = X_eval.fillna(value=0), y_eval.fillna(value=0)

    X_train.drop(labels=correlated_features, axis=1, inplace=True)
    X_eval.drop(labels=correlated_features, axis=1, inplace=True)

    model = lgb.LGBMRegressor(**param)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric='l1',
        early_stopping_rounds=50,
        callbacks=[
            LightGBMPruningCallback(trial, "l1")
        ],  # Add a pruning callback
    )

    preds = model.predict(X_eval)
    mae = mean_absolute_error(y_eval, preds)
    trial.set_user_attr(key="best_booster", value=model)
    return mae

def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


# In[33]:


for i,filename in enumerate(solar):
    path = os.path.join(import_dir, filename)
    input_ds = pd.read_csv(path, index_col='datetime', parse_dates=[0])
    input_ds.index = input_ds.index.sort_values()
    # THE DATA AFTER WHICH THE TEST SERIES STARTS

    input_ds = input_ds.drop('mean_similar', axis=1)

    # Divide features and labels
    input_X, input_y = input_ds.iloc[:,1:], input_ds.iloc[:,0]


    # Create study that minimizes
    study = optuna.create_study(direction="minimize")

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

    train_date = '01-Oct-2020'
    split_date = '01-Nov-2020'

    X_train, y_train = input_X.loc[input_X.index < train_date].copy(), input_y.loc[input_y.index < train_date].copy()
    X_test, y_test = input_X.loc[input_X.index >= split_date].copy(), input_y.loc[input_y.index >= split_date].copy()

    n = len(X_train)

    X_test = X_test.fillna(value=0)
    X_test.drop(labels=correlated_features, axis=1, inplace=True)

    print(f"Optimized MAE: {study.best_value:.5f}")
    seas = 28 * 24 * 4
    h = 2976
    y = input_ds[:'Oct-2020']['value']
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
    forecast = pd.DataFrame(lgbmodel.predict(X_test).clip(0), index=X_test.index)

    #Plotting feature importance
    f_imp, ax_imp = plt.subplots(1, figsize=(30,15))
    plt.gcf().subplots_adjust(left=0.5)
    lgb.plot_importance(lgbmodel, height=0.3, ax=ax_imp)
    plt.savefig(feature_plots_dir+namerun+'_{}.png'.format(os.path.splitext(filename)[0]))

    # FILL ZEROS FOR PLOTTING
    input_ds = input_ds.fillna(value=0)
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    input_ds.loc['Sep-2020':'Oct-2020']['value'].plot(ax=ax, style=['-'])
    forecast.plot(ax=ax, style=['-'])
    ax.set_xbound(lower='2020-09', upper='2020-12')
    plot = plt.suptitle(filename, y=1.01)
    plot = plt.title('November Forecasted')
    plt.savefig(plots_dir+namerun+'_{}.png'.format(os.path.splitext(filename)[0]))
    s = 28 * 24 * 4
    h = 2976
    y = input_ds['value']
    predictions[filename] = list(forecast)
    results_df = pd.concat([results_df, forecast.rename({0:os.path.splitext(filename)[0]}, axis=1).T])


# In[28]:


results_df
results_df.to_csv('./results/solar.csv', header=False)


# In[ ]:
