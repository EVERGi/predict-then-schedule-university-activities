print('_______________________________________')
print('preprocessing starts')
import scripts.preprocessing
print('preprocessing finishes')
print('_______________________________________')
print('forecasting solar')
print('_______________________________________')
import scripts.solar_forecast
print('forecasting solar is succesful')
print('_______________________________________')
print('forecasting buildings 0,1,3')
print('_______________________________________')
import scripts.B013_forecast
print('forecasting buildings 0,1,3 is succesful')
print('_______________________________________')
print('forecasting building 4')
print('_______________________________________')
import scripts.B4_forecast
print('forecasting buildings 4 is succesful')
print('_______________________________________')
print('forecasting buildings 5,6')
print('_______________________________________')
import scripts.B56_forecast
print('forecasting buildings 5,6 is succesful')
print('_______________________________________')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

solar = pd.read_csv('./results/solar.csv', header=None)
b013 = pd.read_csv('./results/buildings_B5B6.csv', header=None)
b4 = pd.read_csv('./results/buildings', header=None)
b56 = pd.read_csv('./results/buildings_18Oct_stationary.csv', header=None)

results = pd.concat([solar, b013, b4, b56], axis=0)
print(results.head())
results = results.sort_values(by=0)
results = results.set_index(0)
results.to_csv('./results/submission_phase2.csv', header=False)
