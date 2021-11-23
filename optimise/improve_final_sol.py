from create_solutions import get_objective_function
from instance_parser import Problem
import datetime
import os
import multiprocessing
import time

def improve_sol(problem: Problem, version, sol_dir, no_once_off = False, no_recur = False):
    
    all_sols = problem.sol_small+problem.sol_large
    all_act = problem.recur+problem.once_off
    
    
    for sol in all_sols:
        if (not sol["recur"] and no_once_off) or (sol["recur"] and no_recur):
            if sol in problem.sol_small:
                problem.sol_small.remove(sol)
            else:
                problem.sol_large.remove(sol)
    

    all_sols = problem.sol_small+problem.sol_large

    act_days_recur = dict()
    act_days_once_off = dict()
    
    for sol in all_sols:
        if sol["recur"]:
            act_days_recur[sol["id"]] = sol["day"]
        else:
            act_days_once_off[sol["id"]] = sol["day"]
    prog_count = 0
    # print("Progress")
    for level in range(0, 5):
        for act in all_act:
            all_sols = problem.sol_small+problem.sol_large
            recur = act in  problem.recur
            if not recur:
                continue
            id = act["id"]
            if recur:
                info = act
                tree = problem.tree_recur[id]
            else:
                info = act
                tree = problem.tree_once_off[id]
            
            small = info["room_size"] == "S"

            precedences = tree["precedences"]

            nodes_below = [node["id"] for node in tree["nodes_below"]]

            if level == tree["levels_above"]:
                prog_count += 1
                print(prog_count,"out of ", len(all_act))
                if recur:
                    day_start = tree["days_allowed"][0]
                    day_end = tree["days_allowed"][-1]+1
                else:
                    if version == 1:
                        day_start = tree["days_allowed"][0]
                        day_end = tree["days_allowed"][-1]+1
                    if version == 2:
                        day_start = 0
                        day_end = 31
                
                count_prec = 0
                for sol in all_sols:
                    if sol["recur"] == recur:
                        if sol["id"] in precedences:
                            count_prec += 1
                            day_start = max(day_start, sol["day"]+1)
                        elif sol["id"] in nodes_below:
                            day_end = min(day_end, sol["day"])
                
                if count_prec != len(precedences):
                    continue

                possible_day_time = list()
                
                for day in range(day_start, day_end):
                    duration = act["duration"]
                    if recur:
                        possible_day_time += [[day, time] for time in range(9*4,17*4-duration)]
                    else:
                        if day == 0:
                            possible_day_time += [[day, time+day*(24*4)] for time in range(0,(24-11)*4)]
                        elif day == 30:
                            possible_day_time += [[day, time+day*(24*4)-11*4] for time in range(0,11*4-duration)]
                        else:
                            possible_day_time += [[day, time+day*(24*4)-11*4] for time in range(0,24*4)]
                
                best_score = problem.get_objective_function()
                
                if recur and no_recur:
                    best_score = 100000

                in_sol = False
                if small:
                    for small_ind, small_sol in enumerate(problem.sol_small):
                        if small_sol["recur"] == recur and small_sol["id"] == id:
                            old_sol = small_sol
                            sol_ind = small_ind
                            in_sol = True
                            break
                else:
                    for large_ind, large_sol in enumerate(problem.sol_large):
                        if large_sol["recur"] == recur and large_sol["id"] == id:
                            old_sol = large_sol
                            sol_ind = large_ind
                            in_sol = True
                            break
                
                
                for poss in possible_day_time:
                    new_sol = {"recur": recur, "id": id, "day": poss[0], "start_time": poss[1], "keep": True}
                
                    if in_sol:
                        if small:
                            problem.sol_small.remove(old_sol)
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_small.insert(sol_ind, new_sol)
                        else:
                            problem.sol_large.remove(old_sol)
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_large.insert(sol_ind, new_sol)
                    else:
                        if small:
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_small += [new_sol]
                        else:
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_large += [new_sol]
                    
                    score = problem.get_objective_function()
                    
                    if score < best_score and feasible:
                        # print(score)
                        best_score = score
                        old_sol = new_sol
                        if not in_sol:
                            in_sol = True
                            sol_ind = -1
                    else:
                        if small:
                            problem.sol_small.remove(new_sol)
                            if in_sol:
                                problem.sol_small.insert(sol_ind, old_sol)
                        else:
                            problem.sol_large.remove(new_sol)
                            if in_sol:
                                problem.sol_large.insert(sol_ind, old_sol)   

    for level in range(0, 31):
        for act in all_act:
            all_sols = problem.sol_small+problem.sol_large
            recur = act in  problem.recur
            if recur:
                continue
            id = act["id"]
            if recur:
                info = act
                tree = problem.tree_recur[id]
            else:
                info = act
                tree = problem.tree_once_off[id]
            
            small = info["room_size"] == "S"

            precedences = tree["precedences"]

            nodes_below = [node["id"] for node in tree["nodes_below"]]

            if level == tree["levels_above"]:
                prog_count += 1
                print(prog_count,"out of ", len(all_act))
                if recur:
                    day_start = tree["days_allowed"][0]
                    day_end = tree["days_allowed"][-1]+1
                else:
                    if version == 1:
                        day_start = tree["days_allowed"][0]
                        day_end = tree["days_allowed"][-1]+1
                    if version == 2:
                        day_start = 0
                        day_end = 31
                
                count_prec = 0
                for sol in all_sols:
                    if sol["recur"] == recur:
                        if sol["id"] in precedences:
                            count_prec += 1
                            day_start = max(day_start, sol["day"]+1)
                        elif sol["id"] in nodes_below:
                            day_end = min(day_end, sol["day"])
                
                if count_prec != len(precedences):
                    continue

                possible_day_time = list()
                
                for day in range(day_start, day_end):
                    duration = act["duration"]
                    if recur:
                        possible_day_time += [[day, time] for time in range(9*4,17*4-duration)]
                    else:
                        if day == 0:
                            possible_day_time += [[day, time+day*(24*4)] for time in range(0,(24-11)*4)]
                        elif day == 30:
                            possible_day_time += [[day, time+day*(24*4)-11*4] for time in range(0,11*4-duration)]
                        else:
                            possible_day_time += [[day, time+day*(24*4)-11*4] for time in range(0,24*4)]
                
                best_score = problem.get_objective_function()
                
                if recur and no_recur:
                    best_score = 100000

                in_sol = False
                if small:
                    for small_ind, small_sol in enumerate(problem.sol_small):
                        if small_sol["recur"] == recur and small_sol["id"] == id:
                            old_sol = small_sol
                            sol_ind = small_ind
                            in_sol = True
                            break
                else:
                    for large_ind, large_sol in enumerate(problem.sol_large):
                        if large_sol["recur"] == recur and large_sol["id"] == id:
                            old_sol = large_sol
                            sol_ind = large_ind
                            in_sol = True
                            break
                
                
                for poss in possible_day_time:
                    new_sol = {"recur": recur, "id": id, "day": poss[0], "start_time": poss[1], "keep": True}
                
                    if in_sol:
                        if small:
                            problem.sol_small.remove(old_sol)
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_small.insert(sol_ind, new_sol)
                        else:
                            problem.sol_large.remove(old_sol)
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_large.insert(sol_ind, new_sol)
                    else:
                        if small:
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_small += [new_sol]
                        else:
                            feasible = problem.assign_rooms(new_sol)
                            problem.sol_large += [new_sol]
                    
                    score = problem.get_objective_function()
                    
                    if score < best_score and feasible:
                        # print(score)
                        best_score = score
                        old_sol = new_sol
                        if not in_sol:
                            in_sol = True
                            sol_ind = -1
                    else:
                        if small:
                            problem.sol_small.remove(new_sol)
                            if in_sol:
                                problem.sol_small.insert(sol_ind, old_sol)
                        else:
                            problem.sol_large.remove(new_sol)
                            if in_sol:
                                problem.sol_large.insert(sol_ind, old_sol)                                 
    
    res = problem.get_objective_function()
    str_sol = problem.create_sol_string()
    instance_path = problem.filepath
    file_path = sol_dir+instance_path.split("/")[-1].replace("phase2_instance_", "sol_").replace(".txt", "")+"_"+str(res)+".txt"
    
    if version == 2:
        with open(file_path, "w+") as file:
            file.write(str_sol)
    return file_path

def process(sol_path, dir_improved, version):

    inst = sol_path.split("_")[-3]+"_"+sol_path.split("_")[-2]
    inst_path = "instances_p2/phase2_instance_"+inst+".txt"
    start = datetime.datetime(2020, 11, 1)
    end = datetime.datetime(2020, 12, 1)
    prob = Problem(inst_path, start, end)
    prob.parse_solution_file(sol_path)

    prev_res = prob.get_objective_function()
    
    new_path = improve_sol(prob, 1, dir_improved, version==2)
    intermediate_res = prob.get_objective_function()
    #print(get_objective_function(inst_path, new_path))
    new_path = improve_sol(prob, 2, dir_improved)

    res = prob.get_objective_function()

    if not os.path.exists("improvement_results.csv"):
        with open("improvement_results.csv", "a+") as file:
            file.write("instance,version,improvement_1,improvement_2")

    improvement_1 = intermediate_res-prev_res

    improvement_2 = res-prev_res
    with open("improvement_results.csv", "a+") as file:
        file.write("\n"+inst+","+str(version)+","+str(improvement_1)+","+str(improvement_2))


def improve_dir(dir_to_impr, dir_improved, version):
    jobs = list()
    done_sols = list()
    if not os.path.exists(dir_improved):
        os.mkdir(dir_improved)
    for i, filepath in enumerate(sorted(os.listdir(dir_to_impr))):
        instance = filepath.split("_")[1]+"_"+filepath.split("_")[2]
        if instance in done_sols:
            continue
        else:
            done_sols.append(instance)
        p=multiprocessing.Process(target=process, args=(dir_to_impr+"/"+filepath, dir_improved, version, ))
        jobs.append(p)
        p.start()
        time.sleep(0.5)

        #if i%10 == 9:
        #    p.join()
    return jobs

def complete_improve():
    all_jobs = list()
    jobs = improve_dir("solutions_sched_p2/", "solutions_impr_p2/", 2)
    all_jobs += jobs
    jobs = improve_dir("solutions_sched_p2/", "solutions_impr_p2/", 1)
    all_jobs += jobs

    for job in all_jobs:
        job.join()

if __name__ == "__main__":
    improve_dir("solutions_sched_p2/", "solutions_impr_p2/", 2)
    improve_dir("solutions_sched_p2/", "solutions_impr_p2/", 1)
    # improve_dir("solutions_sched_p2/", "solutions_impr_p2/", 2)
