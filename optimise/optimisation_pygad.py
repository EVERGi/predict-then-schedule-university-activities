import pygad
import numpy as np
import time
import datetime

from instance_parser import Problem, phase
from create_solutions import create_feasible_schedule, get_objective_function, sol_tree_push, create_feasible_schedule_batt, remove_once_off
import multiprocessing
from optimisation_pygmo import MonashSchedule
import os

batt = False


def do_optimisation(inst_num, pop_num):
    while True:
        gen = 200000
        instances = ["small_" + str(i) for i in range(5)] + ["large_" + str(i) for i in range(5)]
        
        

        if phase == 1:
            instance_dir = "instances/"
            instance_path = instance_dir + "phase1_instance_" + instances[inst_num] + ".txt"

            start = datetime.datetime(2020, 10, 1)
            end = datetime.datetime(2020, 11, 1)

        if phase == 2:
            instance_dir = "instances_p2/"
            instance_path = instance_dir + "phase2_instance_" + instances[inst_num] + ".txt"

            start = datetime.datetime(2020, 11, 1)
            end = datetime.datetime(2020, 12, 1)

        monash = MonashSchedule(instance_path, start, end, instances[inst_num]+",pygad,"+str(pop_num))

        def fitness_func(solution, solution_idx):
            fit_value = -monash.fitness(solution)[0]

            return fit_value

        gen_result_name = "generation_results/"+instances[inst_num]+"_pygad.csv"
        
        if not os.path.exists("generation_results/"):
            os.mkdir("generation_results/")

        if not os.path.exists(gen_result_name):
                with open(gen_result_name, "a+") as file:
                    file.write("population,generation,score")

        def on_generation(ga):
            stop = 500
            best_sol = max(ga.last_generation_fitness)
            
            with open(gen_result_name, "a+") as file:
                file.write("\n"+str(pop_num)+","+str(ga.generations_completed)+ ","+str(best_sol))
                    
            print("Generation: ", ga.generations_completed, ", instance: ", instances[inst_num])
            print("Best solution: ", best_sol)
            last_fits = ga.best_solutions_fitness[-stop:]
            if len(last_fits) == stop and max(last_fits)-min(last_fits) <1:
                return "stop"

        bounds = monash.get_bounds()

        gene_space = [[j for j in range(int(i)+1)] for i in bounds[1]]
        ga_instance = pygad.GA(num_generations=gen,
                            sol_per_pop=pop_num,
                            num_parents_mating=int(pop_num*0.10),
                            fitness_func=fitness_func,
                            gene_type=int,
                            gene_space=gene_space,
                            on_generation=on_generation,
                            num_genes=len(gene_space))

        # Running the GA to optimize the parameters of the function.
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        #print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        #print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

def multi_execute(pop):
    jobs = list()
    for i in range(10):
            p = multiprocessing.Process(target=do_optimisation, args=(i, pop, ))
            p.start()

            jobs.append(p)
            # p.daemon=True
    return jobs

if __name__ == "__main__":

    jobs = []

    if True:
        for i in range(10):
            p = multiprocessing.Process(target=do_optimisation, args=(i, 500, ))
            jobs.append(p)
            p.start()
    else:
        do_optimisation(0)
