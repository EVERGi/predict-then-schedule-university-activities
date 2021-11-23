# EVERGI_predict_optimize

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
pip install -r requirements.txt .
```

The modified Optim_eval needs to be compiled, to do this execute the following commands:

```shell
cd optimise/Optim_eval/
mvn package
```

### Execution

When everything is correctly installed you can execute
The forecast is generated when running execute.py script from the forecast directory
The final output of the forecast is saved to './resutls/submission_phase2.csv'
