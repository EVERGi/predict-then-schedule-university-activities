#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
# os.chdir(os.path.dirname(os.getcwd()))


# In[3]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import seaborn as sns
import matplotlib.pyplot as plt

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


# In[4]:


from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_absolute_error
import pickle
from sklearn.feature_selection import VarianceThreshold
from scipy import stats


# In[5]:


import_dir = './data/lgbm_inputs_phase2/'
# THIS IS THE LIST OF TIMESERIES WHERE NANS ARE FILLED WITH 0 FOR TRAINING
filelist = os.listdir(import_dir)
if ('.ipynb_checkpoints' in filelist):
    filelist.remove('.ipynb_checkpoints')
buildings = ['Building5.csv', 'Building6.csv']
print(list(buildings))
plots_dir = './plots/'
log_dir = './logs/'
model_dir = './models/'
namerun = 'stationary_yearly'
prophets_dir = './data/prophets/'
feature_plots_dir = './plots/importances/'
drop_features = ['y', 'value']
predictions = {}
results_df = pd.DataFrame()
prophets = {}
transform = 'log'


# In[21]:


plt.style.use('fivethirtyeight')


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
    }

    #model = lgb.LGBMRegressor(**param)

    ds_train = lgb.Dataset(X, y, categorical_feature=['working', 'weekday'])
    ds_validate = lgb.Dataset(X_valid, y_valid, categorical_feature=['working', 'weekday'])

    model= lgb.train(
        train_set = ds_train,
        valid_sets=[ds_validate],
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


# In[24]:


for i,filename in enumerate(buildings):
    path = os.path.join(import_dir, filename)
    input_ds = pd.read_csv(path, index_col='datetime', parse_dates=[0])
    input_ds = input_ds.rename({'value':'y'},axis=1)
    input_ds['value'] = input_ds['y']
    input_ds['y'] = np.log1p(input_ds['y'])
    input_ds.index = input_ds.index.sort_values()
    #forecast = prophets[filename]
    df_name = filename + '_' + transform
    #forecast = pd.read_csv(os.path.join(prophets_dir, df_name), index_col='ds', parse_dates=[0])


    if  (filename != 'Building5.csv'):
        input_ds = input_ds.join(input_ds['y'].shift(freq='35D'),how='inner',rsuffix='-3360')

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

    forecast = pd.DataFrame(add_forecast['additive'], index=input_ds_test.index, columns=['additive'])
    forecast = np.expm1(forecast)

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
    predictions[filename] = list(forecast)
    results_df = pd.concat([results_df, forecast.rename({0:os.path.splitext(filename)[0]}, axis=1).T])


# In[26]:


results_df.index = [os.path.splitext(filename)[0] for filename in buildings]
#results_df.drop(labels=['Building3', 'Building4'], axis=0)
results_df.to_csv('./results/buildings_B5B6.csv', header=False)
results_df


# In[ ]:
