# Evolutionary scheduling of university activities based on consumption forecasts to minimise electricity costs
---------------------------------------------------------------------------------------------------------------

This repository contains the code used in the "[Evolutionary scheduling of university activities based on consumption forecasts to minimise electricity costs](https://arxiv.org/abs/2202.12595)" paper presented at the 2022 IEEE Congress on Evolutionary Computation as well as to obtain the 3rd place in the "[IEEE-CIS technical challenge on predict+optimize for renewable energy scheduling](https://ieee-dataport.org/competitions/ieee-cis-technical-challenge-predictoptimize-renewable-energy-scheduling)".

### Installation
---------------

For the scheduling, you will need to get a Gurobi licence and install the  solver as instructed here:

https://www.gurobi.com/documentation/quickstart.html

Then install necessary python libraries by executing:
```shell
pip install -r requirements.txt
```

For the prophet and pygmo library it is better to install them using conda

```shell
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install prophet==1.0.1
conda install pygmo==2.16.0
```

The objective function for the schedule written by the organiser of the competition needs to be build with Maven before executing the shedule optimisation:

```shell
cd optimise/Optim_eval/
mvn package
```

Java needs to be intalled to execute the built jar files in the command line.


## Forecast generation

The forecasting code has been tested successfully on MacOs and Linux with python version 3.8

The forecast is generated when running [execute.py](forecast/execute.py) script from the forecast directory.
First a preprocessing script is run, which takes about a minute.
Further, the scripts for producing forecast for solar and building time series are executed in succession.
An estimate for the running time should be about an hour.
The final output of the forecast is saved to './results/submission_phase2.csv'


### Schedule optimisation

The schedule optimisation code has been tested successfully on Linux with python versions 3.7 and 3.8.

When everything is correctly installed you can execute the [execute.py](optimise/execute.py) file in the optimise folder.

!!! You need to execute it from inside the optimise folder so don't forget to execute the following command when in the root directory of the project:
```shell
cd optimise
```

The execution will run in about 30 minutes, the results shown are obtained running CMA-ES with a population size of 100 for 12 hours. This execution is exactly the same as shown in section n IV-B5 of the paper.
You can change the population size, evolutionary algorithm used and time to find the base solution in the [execute.py](optimise/execute.py) file itself.

The plots that appear show the graphs presented in the paper but for the run you just executed and also the activity plot for the small_0 instance submitted for the competition and presented in the paper. In addition there is also a plot of the final load after improvement (in blue) and of the load with the battery (in orange) for the small_0 instance.
