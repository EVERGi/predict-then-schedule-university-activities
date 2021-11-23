import optimisation_pygad
import optimisation_pygmo
import improve_final_sol
import optimize_battery
import generate_results

import shutil
import time
import os


def get_base_solution(cmaes, pop):
    if cmaes:
        jobs = optimisation_pygmo.multi_execute(pop)
    else:
        jobs = optimisation_pygad.multi_execute(pop)
    
    return jobs

def remove_prev_runs():
    shutil.rmtree('generation_results/', ignore_errors=True)
    shutil.rmtree('solutions_impr_p2/', ignore_errors=True)
    shutil.rmtree('solutions_sched_p2/', ignore_errors=True)
    shutil.rmtree('solutions_batt_p2/', ignore_errors=True)
    silentremove("improvement_battery.csv")
    silentremove("improvement_results.csv")
    silentremove("load_java.csv")
    silentremove("log_results.csv")
    silentremove("load_price.csv")

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
if __name__ == "__main__":
    
    cmaes = True
    pop = 200

    seconds = 120

    remove_previous_runs = False

    if remove_previous_runs:
        remove_prev_runs()


    jobs = get_base_solution(cmaes, pop)
    
    time.sleep(seconds)

    for job in jobs:
        job.terminate()

    improve_final_sol.complete_improve()
    
    optimize_battery.complete_batt_sched()
    
    generate_results.plot_all_results()
    