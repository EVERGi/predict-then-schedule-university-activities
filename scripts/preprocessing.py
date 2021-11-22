#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os


# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from meteocalc import feels_like, heat_index
from workalendar.oceania import Australia


# In[ ]:


# dir = os.chdir(os.path.dirname(os.getcwd()))
# print(os.getcwd())
# print(dir)


# In[ ]:


def solar_slices(import_dir='./data/generation_phase2/', weather_data='./data/weather/weather_all.csv', output_dir='./data/generation_phase2_sliced/'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filelist = os.listdir(import_dir)
    if ('.ipynb_checkpoints' in filelist):
        filelist.remove('.ipynb_checkpoints')
    print(list(filelist))
    for filename in filelist:
        path = os.path.join(import_dir, filename)
        export_path = os.path.join(output_dir, filename)
        df = pd.read_csv(path, index_col='datetime')
        df.index = pd.to_datetime(df.index)
        if filename == 'Solar0.csv':
            split_date = '01-Jun-2020'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)
        if filename == 'Solar1.csv':
            split_date = '01-Oct-2019'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)
        if filename == 'Solar3.csv':
            split_date = '01-Jun-2020'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)
        if filename == 'Solar2.csv':
            split_date = '01-Nov-2019'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)
        if filename == 'Solar5.csv':
            split_date = '01-Oct-2019'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)
        if filename == 'Solar4.csv':
            split_date = '25-Sep-2019'
            df_temp = df[split_date:].copy()
            df_temp.to_csv(export_path)

def c2f(T):
    return T * 9 / 5. + 32

def windchill(T, v):
    return (10*v**.5 - v + 10.5) * (33 - T)


# In[ ]:


def hour(x):
    hm = list(map(int, x.split(' ')[1].split(':')[:2]))
    return hm[0] + 0.5 if hm[1] == 30 else 0


def is_working(date):
    cal = Australia()
    return int(cal.is_working_day(date))


def create_model_ins(import_dir='./data/load_phase2/', weather_data='./data/weather/weather_all.csv', output_dir='./data/xgboost_ins_october/'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filelist = os.listdir(import_dir)
    if ('.ipynb_checkpoints' in filelist):
        filelist.remove('.ipynb_checkpoints')
    print(list(filelist))
    for filename in filelist:
        path = os.path.join(import_dir, filename)
        df = pd.read_csv(path, index_col='datetime')
        df.index = pd.to_datetime(df.index)

        # Add weather features
        weather = pd.read_csv(weather_data, index_col='datetime')
        weather.index = pd.to_datetime(weather.index)
        weather['RH'] = 100 - 5 *             (weather['temperature (degC)'] -
             weather['dewpoint_temperature (degC)'])
        weather['heat'] = weather.apply(lambda x: heat_index(
            c2f(x['temperature (degC)']), x.RH).c, axis=1)
        weather['windchill'] = weather.apply(lambda x: windchill(
            x['temperature (degC)'], x['wind_speed (m/s)']), axis=1)
        weather['feellike'] = weather.apply(lambda x: feels_like(
            c2f(x['temperature (degC)']), x.RH, x['wind_speed (m/s)']*2.237).c, axis=1)

        # Add out-of-sample timestamps
        df = df.join(df.shift(freq='35D'), how='outer', rsuffix='-3360')
        df = df.drop('value-3360', axis=1)
        df = df.join(weather, how='inner')

        # CREATE OCCUPANCY VARIABLE
        occ1 = np.where((df.index > '2020-06-20') &
                        (df.index < '2020-08-02'), 0.1, 0)
        occ2 = np.where((df.index > '2020-08-02') &
                        (df.index < '2020-09-13'), 0.05, 0)
        occ3 = np.where((df.index > '2020-09-13') &
                        (df.index < '2020-10-18'), 0.25, 0)
        occ4 = np.where((df.index > '2020-10-18'), 0.30, 0)
        df['occupancy'] = np.sum([occ1, occ2, occ3, occ4], axis=0)
        X = df.value.groupby(df.index.month).transform('mean')
        X_scaled = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        df['occupancy'] = np.sum([df.occupancy, X_scaled], axis=0)


        # Add calendar variables
        idx_df = pd.DataFrame(df.index, index=df.index)
        cos_dayofyear = idx_df.applymap(lambda x: x.dayofyear).applymap(
            lambda x: np.cos(x/365)).rename({'datetime': 'cos-dayofyear'}, axis=1)
        sin_dayofyear = idx_df.applymap(lambda x: x.dayofyear).applymap(
            lambda x: np.cos(x/365)).rename({'datetime': 'sin-dayofyear'}, axis=1)
        cos_month = idx_df.applymap(lambda x: x.month).applymap(
            lambda x: np.cos(x/12)).rename({'datetime': 'cos-month'}, axis=1)
        sin_month = idx_df.applymap(lambda x: x.month).applymap(
            lambda x: np.sin(x/12)).rename({'datetime': 'sin-month'}, axis=1)

            # DAY OF WEEK DUMMIES
        dow = idx_df.applymap(lambda x: x.weekday())
        dow = pd.Categorical(
            dow, categories=[0, 1, 2, 3, 4, 5, 6], ordered=True)
        dow = idx_df.applymap(lambda x: x.weekday()).rename(
            {'datetime': 'weekday'}, axis=1)

        cos_hour = idx_df.applymap(lambda x: x.hour).applymap(
            lambda x: np.cos(x/24)).rename({'datetime': 'cos-hour'}, axis=1)
        sin_hour = idx_df.applymap(lambda x: x.hour).applymap(
            lambda x: np.sin(x/24)).rename({'datetime': 'sin-hour'}, axis=1)

            # WORKING DAY TICKER
        working = idx_df.applymap(lambda x: is_working(
            x)).rename({'datetime': 'working'}, axis=1)
        # Concat everything together
        df = pd.concat([df, cos_month, sin_month, cos_dayofyear,
                       sin_dayofyear, dow, cos_hour, sin_hour, working], axis=1)
        df = df[:'2020-11']
        #print(df.columns)
        # print(df.tail()['cos-dayofyear'])
        df.to_csv(output_dir+filename)


# In[ ]:


def create_solar_model_ins(import_dir='./data/generation_phase2_sliced/', weather_data='./data/weather/weather_all.csv', output_dir='./data/xgboost_ins_october/'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    filelist = os.listdir(import_dir)
    if ('.ipynb_checkpoints' in filelist):
        filelist.remove('.ipynb_checkpoints')
    print(list(filelist))
    for filename in filelist:
        path = os.path.join(import_dir, filename)
        df = pd.read_csv(path, index_col='datetime')
        df.index = pd.to_datetime(df.index)

        # Add weather features
        weather = pd.read_csv(weather_data, index_col='datetime')
        weather.index = pd.to_datetime(weather.index)
        # Add out-of-sample timestamps
        df = df.join(df.shift(freq='35D'), how='outer', rsuffix='-3360')
        df = df.drop('value-3360', axis=1)

        # Add mean value at a similar time feature
        df['mean_similar'] = df.value.groupby(df.index.time).transform('mean')
        df = df.join(weather, how='inner')
        #print(df.head())

        idx_df = pd.DataFrame(df.index, index=df.index)
        #date = lambda x: datetime.date.fromisoformat(x.split(' ')[0])
        cos_dayofyear = idx_df.applymap(lambda x: x.dayofyear).applymap(
            lambda x: np.cos(x/365)).rename({'datetime': 'cos-dayofyear'}, axis=1)
        sin_dayofyear = idx_df.applymap(lambda x: x.dayofyear).applymap(
            lambda x: np.cos(x/365)).rename({'datetime': 'sin-dayofyear'}, axis=1)
        cos_month = idx_df.applymap(lambda x: x.month).applymap(
            lambda x: np.cos(x/12)).rename({'datetime': 'cos-month'}, axis=1)
        sin_month = idx_df.applymap(lambda x: x.month).applymap(
            lambda x: np.sin(x/12)).rename({'datetime': 'sin-month'}, axis=1)

        cos_hour = idx_df.applymap(lambda x: x.hour).applymap(
            lambda x: np.cos(x/24)).rename({'datetime': 'cos-hour'}, axis=1)
        sin_hour = idx_df.applymap(lambda x: x.hour).applymap(
            lambda x: np.sin(x/24)).rename({'datetime': 'sin-hour'}, axis=1)


        df = pd.concat([df, cos_month, sin_month, cos_dayofyear,
                       sin_dayofyear, cos_hour, sin_hour], axis=1)
        df = df[:'2020-11']
        #print(df.columns)
        df.to_csv(output_dir+filename)


# In[ ]:


solar_slices(import_dir='./data/generation_phase2/')


# In[ ]:


create_model_ins(weather_data='./data/weather/weather_combi.csv',
                 output_dir='./data/lgbm_inputs_phase2/')


# In[ ]:


create_solar_model_ins(weather_data='./data/weather/weather_combi.csv',
                       import_dir='./data/generation_phase2_sliced/', output_dir='./data/lgbm_inputs_phase2/')
