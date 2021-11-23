import optimisation_pygad
import optimisation_pygmo
import improve_final_sol
import optimize_battery

import multiprocessing
import time

def get_base_solution(cmaes, pop):
    if cmaes:
        jobs = optimisation_pygmo.multi_execute(pop)
    else:
        jobs = optimisation_pygad.multi_execute(pop)
    
    return jobs

def improve_solution():
    pass

def battery_schedule():
    pass

def plot_results():
    pass

if __name__ == "__main__":
    
    cmaes = False
    pop = 200

    seconds = 30
    jobs = get_base_solution(cmaes, 200)
    
    time.sleep(seconds)

    for job in jobs:
        job.terminate()

    improve_final_sol.complete_improve()
    
    optimize_battery.complete_batt_sched()

