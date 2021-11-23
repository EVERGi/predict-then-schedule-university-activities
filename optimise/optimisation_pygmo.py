import pygmo as pg
import numpy as np
import time
import datetime

from instance_parser import Problem, phase, version
from create_solutions import create_feasible_schedule, get_objective_function, sol_tree_push, create_feasible_schedule_batt, remove_once_off
import multiprocessing
from io import StringIO
import sys
import time
import threading

import os

batt = False

class MonashSchedule:
    def __init__(self, filepath, start, end, info):

        self.prob = Problem(filepath, start, end)
        self.tree_once_off = self.prob.tree_once_off
        self.tree_recur = self.prob.tree_recur
        self.info = info

        self.trial = -1

        epsilon = 0.00001
        if version == 1:
            self.bnds_recur = [len(i["days_allowed"])-epsilon for i in self.tree_recur]
            self.bnds_once_off = [len(i["days_allowed"])-epsilon for i in self.tree_once_off]

            self.bnds_time_recur = [(17-9)*4-i["duration"]-epsilon for i in self.prob.recur]
            self.bnds_time_once_off = [24*4-epsilon for _ in self.prob.once_off]
        elif version == 2:
            self.bnds_recur = [len(i["days_allowed"])*(17-9)*4-epsilon for i in self.tree_recur]
            self.bnds_once_off = [len(self.prob.datetimes)-epsilon for _ in self.tree_once_off]
            
            self.bnds_time_recur = []
            self.bnds_time_once_off = []

        if batt:
            num_bat_var = len(self.prob.datetimes)*len(self.prob.batt)
            self.bnds_batt = [3-epsilon for _ in range(num_bat_var)]

            self.bounds_up = self.bnds_recur+self.bnds_once_off+self.bnds_time_recur+self.bnds_time_once_off+self.bnds_batt
        else:
            self.bounds_up = self.bnds_recur + self.bnds_once_off + self.bnds_time_recur + self.bnds_time_once_off
        self.bounds_low = [0]*len(self.bounds_up)


        self.dim = len(self.bounds_up)
        self.best_result = 10000000000

        if phase == 1:
            self.instance = self.prob.filepath.split("/")[-1].replace("phase1_instance_", "").replace(".txt", "")
        
        if phase == 2:
            self.instance = self.prob.filepath.split("/")[-1].replace("phase2_instance_", "").replace(".txt", "")

    def reset(self):
        self.prob.sol_small = list()
        self.prob.sol_large = list()

        self.sol_batt = list()

    def fitness(self, x):
        self.trial += 1

        self.a = x
        sol = np.floor(x).astype(int)
        if version == 1:
            sol_recur = [self.prob.tree_recur[ind]["days_allowed"][choice] for ind, choice in enumerate(sol[:len(self.bnds_recur)])]
            days_limit = len(self.bnds_recur+self.bnds_once_off)
            sol_once_off = [self.prob.tree_once_off[ind]["days_allowed"][choice] for ind, choice in enumerate(sol[len(self.bnds_recur):days_limit])]

            # print(self.prob.num_weekdays)
            sol_recur = sol_tree_push(self.tree_recur, sol_recur, 5)
            sol_once_off = sol_tree_push(self.tree_once_off, sol_once_off, self.prob.num_all_days)

            end_sol_time_recur = days_limit+len(self.bnds_time_recur)
            sol_time_recur = [9*4+i for i in sol[days_limit:end_sol_time_recur]]
            end_sol_time = end_sol_time_recur+len(self.bnds_time_recur)
            #sol_time_once_off = [9 * 4 + i for i in sol[end_sol_time_recur:end_sol_time]]
            sol_time_once_off = [i for i in sol[end_sol_time_recur:end_sol_time]]
        
            # For australian time
            if sol_time_once_off[0] < 11*4:
                sol_time_once_off[0] = 11*4
            if sol_time_once_off[-1] >= 11*4:
                sol_time_once_off[-1] = 11*4-1

        
        elif version == 2:
            sol_recur = [self.prob.tree_recur[ind]["days_allowed"][time//((17-9)*4)] for ind, time in enumerate(sol[:len(self.bnds_recur)])]
            days_limit = len(self.bnds_recur+self.bnds_once_off)
            sol_once_off = [(time+11*4)//(24*4) for ind, time in enumerate(sol[len(self.bnds_recur):days_limit])]

            # print(self.prob.num_weekdays)
            sol_recur = sol_tree_push(self.tree_recur, sol_recur, 5)
            sol_once_off = sol_tree_push(self.tree_once_off, sol_once_off, self.prob.num_all_days)

            end_sol_time_recur = days_limit+len(self.bnds_time_recur)
            sol_time_recur = [9*4+i%((17-9)*4) for i in sol[:len(self.bnds_recur)]]
            
            sol_time_once_off = [(time+11*4)%(24*4) for ind, time in enumerate(sol[len(self.bnds_recur):days_limit])]


        
        try:
            create_feasible_schedule(self.prob, sol_recur, sol_once_off, sol_time_recur, sol_time_once_off)
        except TypeError:
            self.reset()
            print(200000)
            return [200000]
        

        if batt:
            batt_sol = list()

            start = end_sol_time
            for i in enumerate(self.prob.batt):
                end = start + len(self.prob.datetimes)
                batt_sol.append(sol[start:end])
                start = end

            create_feasible_schedule_batt(self.prob, batt_sol)
        remove_once_off(self.prob)

        sol_string = self.prob.create_sol_string()

        solution_path = str(time.time())+".tmp"

        if False:
            with open(solution_path, "w+") as file:
                file.write(sol_string)
            
            res = get_objective_function(self.prob.filepath, solution_path)
            print(res)

            os.remove(solution_path)

        res = self.prob.get_objective_function()

        if phase == 1:
            solution_dir = "solutions_sched/"
        elif phase == 2:
            solution_dir = "solutions_sched_p2/"
            if not os.path.exists(solution_dir):
                os.mkdir(solution_dir)


        better_sol = False
        start_file = "sol_" + self.instance+"_"
        solution_path = solution_dir + start_file + str(res) + ".txt"
        for file in os.listdir(solution_dir):
            if file.startswith(start_file):
                file_res = float(file.replace(start_file, "").replace(".txt", ""))

                if res+10 > file_res:
                    better_sol = True

        if not better_sol:
            
            with open("log_results.csv", "a+") as text_file:
                
                text_file.write("\n"+self.info+","+str(self.trial)+","+str(res))

            with open(solution_path, "w+") as text_file:
                text_file.write(sol_string)

        self.reset()
        return [res]

    def get_bounds(self):

        return (self.bounds_low, self.bounds_up)


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

        monash = MonashSchedule(instance_path, start, end, instances[inst_num]+",cmaes,"+str(pop_num))
        prob = pg.problem(monash)

        algo = pg.algorithm(pg.cmaes(gen=gen, force_bounds=True, ftol=100, xtol=1))

        pop = pg.population(prob, pop_num)
        algo.set_verbosity(1)

        x = threading.Thread(target=evolve, args=(algo, pop, ))
        x.daemon=True
        x.start()

        prev_length = 0

        stop = 100
        gen_result_name = "generation_results/"+instances[inst_num]+"_pygmo.csv"
        
        if not os.path.exists("generation_results/"):
            os.mkdir("generation_results/")
        if not os.path.exists(gen_result_name):
            with open(gen_result_name, "a+") as file:
                file.write("population,generation,score")
        while True:
            time.sleep(30)
            if not x.is_alive():
                break
            uda = algo.extract(pg.cmaes)
            log = uda.get_log()
            new_length = len(log)
            if new_length != prev_length:
                
                with open(gen_result_name, "a+") as file:
                    for ind in range(prev_length, new_length):
                        file.write("\n"+str(pop_num)+","+str(ind)+ ","+str(log[ind][2]))

                best_sols = [val[2]  for val in log[-stop:]]
                prev_length = new_length

def evolve(algo, pop):
    algo.evolve(pop)
    print("finnished")

def multi_execute(pop):
    jobs = list()
    for i in range(10):
            p = multiprocessing.Process(target=do_optimisation, args=(i, pop, ))
            p.start()

            jobs.append(p)
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
