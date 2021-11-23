# EVERGI_predict_optimize


### Installation
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

After this install the prophet python library. In our case pip returned an error and we had to install it using conda. 

```shell
conda install prophet
```

The modified Optim_eval needs to be compiled, to do this execute the following commands:

```shell
cd optimise/Optim_eval/
mvn package
```

Java also needs to be intalled to execute the build jar files in the command line.


## Forecast generation

The forecasting code has been tested successfully on MacOs with python version 3.8

The forecast is generated when running execute.py script from the forecast directory.
First a preprocessing script is run, which takes about a minute.
Further, the scripts for producing forecast for solar and building time series are executed in succession.
An estimate for the running time should be about an hour.
The final output of the forecast is saved to './results/submission_phase2.csv'


### Schedule optimisation

The optimisation code has been tested successfully on Linux with python versions 3.7 and 3.8.

When everything is correctly installed you can execute the execute.py file in the optimise folder.
!!! You need to execute it from inside the optimis folder so don't forget to execute the following command when in the root directory of the project:
```shell
cd optimise
```

The execution will run in about 30 minutes, the results shown are obtained running CMA-ES with a population size of 200 for 120 seconds. The improved solution and the battery schedule are generated exactly as for the competition.
You can change the population size, evolutionary algorithm used and time to find the base solution in the execute.py file itself.

The plots that appear show the graphs presented in the report but for the run you just executed and also the activity plot for the small_0 instance submitted for the competition and presented in the report. In addition there is also a plot of the final load after improvement (in blue) and of the load with the battery (in orange) for the small_0 instance.