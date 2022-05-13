import os

from instance_parser import Problem, phase
import datetime
import pyomo.environ as pyo
import csv
import numpy as np
from create_solutions import get_objective_function
import multiprocessing
import time

load_price = "load_java.csv"


class ModelBattery:
    def __init__(self, solution_path, instance: Problem):
        self.sol_path = solution_path
        self.load, self.price = get_load_and_price()

        self.instance = instance

        self.batt = instance.batt

        self.model = pyo.ConcreteModel()
        build_sets(self.model, self.instance, self.batt)
        build_params(self.model, self.batt, self.load, self.price)
        build_vars(self.model)
        build_constr(self.model)

        build_obj(self.model)

    def optimise(self):
        opt = pyo.SolverFactory("gurobi")
        opt.solve(self.model, options={"TimeLimit": 1200})

        results = dict()

        for i, batt in enumerate(self.model.batt_id):
            results[i] = list()
            for time in self.model.time:
                value_charge = pyo.value(self.model.charge_batt[time, batt])
                value_discharge = pyo.value(self.model.discharge_batt[time, batt])
                if value_charge >= 0.99:
                    results[i].append(0)
                elif value_discharge >= 0.99:
                    results[i].append(2)
                else:
                    results[i].append(1)
        if False:
            with open("final_load.csv", "w+") as file:

                for time in self.model.time:
                    value_charge = pyo.value(self.model.final_load[time])
                    file.write(str(value_charge) + "\n")

        return results


def build_sets(model, instance, batt):
    time = np.array([i for i, _ in enumerate(instance.datetimes)])
    time_plus = np.array([i for i, _ in enumerate(instance.datetimes + [0])])
    model.time = pyo.Set(initialize=time)
    model.time_plus = pyo.Set(initialize=time_plus)

    batteries = np.array([i for i, _ in enumerate(batt)])
    model.batt_id = pyo.Set(initialize=batteries)


def build_params(model, batt, load, price):
    model.baseload = pyo.Param(model.time, initialize=array_to_dict(load))

    model.price = pyo.Param(model.time, initialize=array_to_dict(price))

    batt_eff = np.array([i["efficiency"] ** 0.5 for i in batt])
    model.batt_eff = pyo.Param(model.batt_id, initialize=array_to_dict(batt_eff))

    batt_max_p = np.array([i["max_power"] for i in batt])
    model.batt_max_p = pyo.Param(model.batt_id, initialize=array_to_dict(batt_max_p))

    batt_cap = np.array([i["capacity"] * 4 for i in batt])
    model.batt_cap = pyo.Param(model.batt_id, initialize=array_to_dict(batt_cap))


def build_vars(model):
    model.charge_batt = pyo.Var(model.time, model.batt_id, domain=pyo.Binary)
    model.discharge_batt = pyo.Var(model.time, model.batt_id, domain=pyo.Binary)

    model.soc = pyo.Var(
        model.time_plus, model.batt_id, domain=pyo.NonNegativeIntegers, initialize=0
    )

    model.final_load = pyo.Var(model.time, domain=pyo.Reals)

    model.max_peak = pyo.Var(domain=pyo.NonNegativeReals)

    model.total_cost = pyo.Var(domain=pyo.Reals)


def build_constr(model):

    # State of charge
    def soc_change(model, t, batt):
        return (
            model.soc[(t + 1), batt]
            == model.soc[t, batt]
            - model.charge_batt[t, batt] * model.batt_max_p[batt]
            + model.discharge_batt[t, batt] * model.batt_max_p[batt]
        )

    model.soc_change = pyo.Constraint(model.time, model.batt_id, rule=soc_change)

    def max_soc(model, t, batt):
        return model.soc[t, batt] <= model.batt_cap[batt]

    model.max_soc = pyo.Constraint(model.time_plus, model.batt_id, rule=max_soc)

    def charge_or_discharge(model, t, batt):
        return model.charge_batt[t, batt] + model.discharge_batt[t, batt] <= 1

    model.charge_or_discharge = pyo.Constraint(
        model.time, model.batt_id, rule=charge_or_discharge
    )

    def final_load_calc(model, t):
        return model.final_load[t] == model.baseload[t] - sum(
            model.discharge_batt[t, batt]
            * model.batt_eff[batt]
            * model.batt_max_p[batt]
            for batt in model.batt_id
        ) + sum(
            model.charge_batt[t, batt] / model.batt_eff[batt] * model.batt_max_p[batt]
            for batt in model.batt_id
        )

    model.final_load_calc = pyo.Constraint(model.time, rule=final_load_calc)

    def max_peak_calc(model, t):
        return model.max_peak >= model.final_load[t]

    model.max_peak_calc = pyo.Constraint(model.time, rule=max_peak_calc)

    def total_cost_calc(model):
        return (
            model.total_cost
            == sum(
                0.25 / 1000 * model.price[t] * model.final_load[t] for t in model.time
            )
            + 0.005 * model.max_peak * model.max_peak
        )

    model.total_cost_calc = pyo.Constraint(rule=total_cost_calc)


def build_obj(model):
    def obj_cost_rule(model):
        return model.total_cost

    model.obj = pyo.Objective(rule=obj_cost_rule, sense=pyo.minimize)


def get_load_and_price():
    with open(load_price, "r+") as csvfile:
        spamreader = csv.reader(csvfile)

        load = list()
        price = list()
        for row in spamreader:
            load.append(float(row[0]))
            price.append(float(row[1]))
        return np.array(load), np.array(price)


def append_opt_to_battery(prev_solution_path, results):
    result_string = "\n"
    for batt_id, result_list in results.items():
        for time_id, value in enumerate(result_list):
            if value != 1:
                result_string += (
                    "c " + str(batt_id) + " " + str(time_id) + " " + str(value) + "\n"
                )

    with open(prev_solution_path, "r+") as file:
        previous_sol = file.read()

    new_sol = previous_sol + result_string

    solution_path = "temp" + str(time.time())

    with open(solution_path, "w+") as file:
        file.write(new_sol)

    return solution_path


def optimise_battery(solution_path, instance, batt_sol_path):
    model = ModelBattery(solution_path, instance)
    result = model.optimise()
    file_path = append_opt_to_battery(solution_path, result)
    instance_path = instance.filepath

    res = get_objective_function(instance_path, file_path)
    print("Result after optimisation")
    print(res)

    if phase == 1:
        sol_path = (
            batt_sol_path
            + instance_path.split("/")[-1]
            .replace("phase1_instance_", "sol_")
            .replace(".txt", "")
            + "_"
            + str(res)
            + ".txt"
        )
    elif phase == 2:
        sol_path = (
            batt_sol_path
            + instance_path.split("/")[-1]
            .replace("phase2_instance_", "sol_")
            .replace(".txt", "")
            + "_"
            + str(res)
            + ".txt"
        )

    os.rename(file_path, sol_path)

    return res


def array_to_dict(array):
    dict_ret = dict()

    for i, elem in enumerate(array):
        dict_ret[i] = elem
    return dict_ret


def multiproc_optimise(sol_dir, to_dir):
    jobs = list()
    for file in os.listdir(sol_dir):
        if phase == 1:
            start = datetime.datetime(2020, 10, 1)
            end = datetime.datetime(2020, 11, 1)
        if phase == 2:
            start = datetime.datetime(2020, 11, 1)
            end = datetime.datetime(2020, 12, 1)

        split_filename = file.split("_")
        instance = split_filename[1] + "_" + split_filename[2]
        if phase == 1:
            instance_path = "instances/phase1_instance_" + instance + ".txt"
        if phase == 2:
            instance_path = "instances_p2/phase2_instance_" + instance + ".txt"

        instance = Problem(instance_path, start, end)

        res = get_objective_function(instance_path, sol_dir + file)
        print("Result before optimisation")
        print(res)
        p = multiprocessing.Process(
            target=optimise_battery,
            args=(
                sol_dir + file,
                instance,
                to_dir,
            ),
        )
        jobs.append(p)
        p.start()
        time.sleep(0.5)
        # optimise_battery(sol_dir+file, instance, to_dir)


def optimise_all_found_sols(sol_dir, to_dir):
    if not os.path.exists("improvement_battery.csv"):
        with open("improvement_battery.csv", "a+") as file:
            file.write("instance,improvement")

    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    done_sols = list()
    for file in sorted(os.listdir(sol_dir)):
        print(file)

        split_filename = file.split("_")
        instance = split_filename[1] + "_" + split_filename[2]

        if instance in done_sols:
            continue
        else:
            done_sols.append(instance)

        if phase == 1:
            start = datetime.datetime(2020, 10, 1)
            end = datetime.datetime(2020, 11, 1)
        if phase == 2:
            start = datetime.datetime(2020, 11, 1)
            end = datetime.datetime(2020, 12, 1)

        if phase == 1:
            instance_path = "instances/phase1_instance_" + instance + ".txt"
        if phase == 2:
            instance_path = "instances_p2/phase2_instance_" + instance + ".txt"

        instance = Problem(instance_path, start, end)

        res = get_objective_function(instance_path, sol_dir + file)

        print("Result before optimisation")
        print(res)

        res_after = optimise_battery(sol_dir + file, instance, to_dir)

        instance = split_filename[1] + "_" + split_filename[2]

        improvement = res_after - res
        with open("improvement_battery.csv", "a+") as file:
            file.write("\n" + instance + "," + str(improvement))


def count_total_score(sol_dir):

    solutions = dict()
    for file in os.listdir(sol_dir):

        split_filename = file.split("_")
        instance = split_filename[1] + "_" + split_filename[2]

        value = float(file.replace("sol_" + instance + "_", "").replace(".txt", ""))

        if instance not in solutions.keys():
            solutions[instance] = value
        elif value < solutions[instance]:
            solutions[instance] = value

    sum = 0
    for key, value in solutions.items():
        sum += value
    # print("Total score")
    # print(sum)
    return sum


def complete_batt_sched():
    sol_dir = "solutions_impr_p2/"
    to_dir = "solutions_batt_p2/"

    optimise_all_found_sols(sol_dir, to_dir)


if __name__ == "__main__":
    sol_dir = "solutions_impr_p2/"
    to_dir = "solutions_batt_p2/"

    optimise_all_found_sols(sol_dir, to_dir)
    # count_total_score(sol_dir)
