# EVERGI_predict_optimize

## Forecast generation
The python version

Install the necessary python libraries by executing:
```shell
pip install -r requirements.txt
```

The forecast is generated when running execute.py script from the forecast directory.
First a preprocessing script is run, which takes about a minute.
Further, the scripts for producing forecast for solar and building time series are executed in succession.
An estimate for the running time should be about an hour.
The final output of the forecast is saved to './results/submission_phase2.csv'

## Schedule optimisation
-------------------------

### Instalation
---------------

First install the C++ dependencies.

The first is pagmo2 and the complete installation instructions can be found on the following link

https://esa.github.io/pagmo2/install.html

Secondly get a licence and install the Gurobi solver as instructed here:

https://www.gurobi.com/documentation/9.0/quickstart_linux/software_installation_guid.html

Then install necessary python libraries by executing:
```shell
pip install -r requirements.txt
```

The modified Optim_eval needs to be compiled, to do this execute the following commands:

```shell
cd optimise/Optim_eval/
mvn package
```

### Execution

When everything is correctly installed you can execute.
