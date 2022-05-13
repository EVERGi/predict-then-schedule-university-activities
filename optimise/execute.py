import optimisation_pygad
import optimisation_pygmo
import improve_final_sol
import optimize_battery
import generate_results

import shutil
import time
import os
import matplotlib.pyplot as plt
import csv
import subprocess

from optimize_battery import count_total_score


def get_base_solution(cmaes, pop):
    if cmaes:
        jobs = optimisation_pygmo.multi_execute(pop)
    else:
        jobs = optimisation_pygad.multi_execute(pop)

    return jobs


def remove_prev_runs():
    shutil.rmtree("generation_results/", ignore_errors=True)
    shutil.rmtree("solutions_impr_p2/", ignore_errors=True)
    shutil.rmtree("solutions_sched_p2/", ignore_errors=True)
    shutil.rmtree("solutions_batt_p2/", ignore_errors=True)
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


def get_number_iterations(library):
    res_dir = "generation_results/"
    iter_small = 0
    iter_large = 0
    for file in os.listdir(res_dir):
        iterations = 0
        if file.find(library) != -1:
            with open(res_dir + file) as csvfile:
                csvreader = csv.reader(csvfile)
                for i, row in enumerate(csvreader):
                    if i != 0:
                        iterations += float(row[0])
            if file.find("small") != -1:
                iter_small += iterations
            else:
                iter_large += iterations
    return {"iter_small": iter_small, "iter_large": iter_large}


def get_total_score_folder(folder, forecasted_data=True):

    total_cost = 0
    for file in os.listdir(folder):
        split_filename = file.split("_")
        instance = split_filename[1] + "_" + split_filename[2]
        if forecasted_data:
            jar_file = "Optim_eval/target/evaluate_instance.jar"
        else:
            jar_file = "Optim_eval/target/evaluate_instance_true.jar"

        instance_path = "instances_p2/phase2_instance_" + instance + ".txt"
        solution_path = folder + "/" + file
        a = subprocess.check_output(
            ["java", "-jar", jar_file, instance_path, solution_path]
        ).decode("utf-8")
        # print(a)
        b = a.split("\n")
        res = float(b[-2])
        total_cost += res

    return total_cost


if __name__ == "__main__":

    cmaes = True
    pop = 100

    seconds = 12 * 60 * 60

    remove_previous_runs = True

    if remove_previous_runs:
        remove_prev_runs()

    start_time = time.time()

    jobs = get_base_solution(cmaes, pop)

    time.sleep(seconds)

    for job in jobs:
        job.terminate()

    base_sched_time = time.time()

    improve_final_sol.complete_improve()

    improve_sched_time = time.time()

    optimize_battery.complete_batt_sched()

    final_sched_time = time.time()

    generate_results.plot_all_results()

    print("\n\n\n=====Summary======\n")
    print("Execution time")
    print("Base schedule: ", base_sched_time - start_time, " sec")
    print("Improved schedule: ", improve_sched_time - base_sched_time, " sec")
    print("Final schedule: ", final_sched_time - improve_sched_time, " sec")
    print("Toal time schedule: ", final_sched_time - start_time, " sec")
    if cmaes:
        iterations = get_number_iterations("pygmo")
    else:
        iterations = get_number_iterations("pygad")
    print("\nSmall instances:")
    print("Average iterations for base schedule:", iterations["iter_small"] / 5)

    print("\nLarge instances:")
    print("Average iterations for base schedule:", iterations["iter_large"] / 5)

    total_cost = count_total_score("solutions_batt_p2")
    print("\n\n Best final cost (including previous runs): ", total_cost)

    plt.show()
