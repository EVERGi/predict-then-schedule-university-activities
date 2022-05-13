import csv
import datetime
import pandas as pd
import copy
import pytz
from visualise_sol import visualise_activities

phase = 2

version = 2

if phase == 1:
    load_price = "Optim_eval/data/load_price_phase_1.csv"
if phase == 2:
    load_price = "Optim_eval/data/load_price_phase_2.csv"


class Problem:
    def __init__(self, filepath, start, end):
        self.filepath = filepath

        self.ppoi = dict()
        self.build = list()
        self.solar = list()
        self.batt = list()
        self.recur = list()
        self.once_off = list()

        self.parse_instance()

        self.small_rooms = self.list_rooms(small=True)
        self.large_rooms = self.list_rooms(small=False)

        self.datetimes_utc = perdelta(start, end, datetime.timedelta(minutes=15))
        tz_utc = pytz.timezone("UTC")
        tz_aedt = pytz.timezone("Australia/Melbourne")
        self.datetimes = list()

        for utc_dt in self.datetimes_utc:
            utc_localised = utc_dt.replace(tzinfo=tz_utc)
            aedt_localized = utc_localised.astimezone(tz_aedt)
            self.datetimes.append(aedt_localized)
        # print(self.datetimes[0])

        self.prices = self.get_prices()
        self.base_load = self.get_base_load()
        self.monday_index = first_monday_index(self.datetimes)
        # print(self.monday_index)
        # print(self.monday_index)

        # self.weekdays = get_weekdays(self.datetimes)
        self.all_days = get_all_days(self.datetimes)

        # self.num_weekdays = len(self.weekdays)
        self.num_all_days = len(self.all_days)

        # print(self.num_weekdays)

        self.tree_recur = create_tree_problem(self.recur)
        set_possible_days(self.tree_recur, 5)
        for info_tree in self.tree_recur:
            precedences = list()
            get_all_precedence_index(info_tree["id"], self.tree_recur, precedences)
            info_tree["all_precedences"] = precedences

        self.tree_once_off = create_tree_problem(self.once_off)
        set_possible_days(self.tree_once_off, self.num_all_days)

        for info_tree in self.tree_once_off:
            precedences = list()
            get_all_precedence_index(info_tree["id"], self.tree_once_off, precedences)
            info_tree["all_precedences"] = precedences

        # set_possible_days(self.tree_once_off, self.num_weekdays)

        # print(self.tree_once_off)
        self.sol_small = list()
        self.sol_large = list()

        self.sol_batt = list()

    def parse_instance(self):

        with open(self.filepath) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=" ")

            for row in spamreader:
                row = [row[0]] + [
                    int_or_float(i) if i not in ["S", "L"] else i for i in row[1:]
                ]
                if row[0] == "ppoi":
                    self.ppoi = parse_ppoi(row)
                elif row[0] == "b":
                    self.build.append(parse_build(row))
                elif row[0] == "s":
                    self.solar.append(parse_solar(row))
                elif row[0] == "c":
                    self.batt.append(parse_batt(row))
                elif row[0] == "r":
                    self.recur.append(parse_recur(row))
                elif row[0] == "a":
                    self.once_off.append(parse_once_off(row))

    def list_rooms(self, small=True):
        list_rooms = list()
        for building in self.build:
            if small:
                list_rooms += [building["id"]] * building["num_small"]
            else:
                list_rooms += [building["id"]] * building["num_big"]
        return list_rooms

    def create_sol_string(self):

        sol_string = ""

        sol_string += (
            "ppoi "
            + str(self.ppoi["num_build"])
            + " "
            + str(self.ppoi["num_solar"])
            + " "
            + str(self.ppoi["num_batt"])
            + " "
            + str(self.ppoi["num_recur"])
            + " "
            + str(self.ppoi["num_once_off"])
            + "\n"
        )

        num_once_off = count_keep(self.sol_small + self.sol_large)

        sol_string += (
            "sched " + str(self.ppoi["num_recur"]) + " " + str(num_once_off) + "\n"
        )

        strings_recur = [""] * self.ppoi["num_recur"]
        strings_once_off = [""] * self.ppoi["num_once_off"]
        for sol in self.sol_small:
            # print(sol)
            if sol["recur"]:
                strings_recur[sol["id"]] = self.activity_sol_to_string(sol, True)
            else:
                if sol["keep"]:
                    strings_once_off[sol["id"]] = self.activity_sol_to_string(sol, True)

        for sol in self.sol_large:
            if sol["recur"]:
                strings_recur[sol["id"]] = self.activity_sol_to_string(sol, False)
            else:
                if sol["keep"]:
                    strings_once_off[sol["id"]] = self.activity_sol_to_string(
                        sol, False
                    )
        kept_o_o = [i for i in strings_once_off if i != ""]

        sol_string += "\n".join(strings_recur) + "\n"
        sol_string += "\n".join(kept_o_o)

        for batt in self.sol_batt:
            sol_string += "\n"
            sol_string += (
                "c "
                + str(batt["id"])
                + " "
                + str(batt["time"])
                + " "
                + str(batt["value"])
            )
        return sol_string

    def activity_sol_to_string(self, sol, small):

        if sol["recur"]:
            start_line = "r "
        else:
            start_line = "a "

        recur_add = self.monday_index
        once_off_add = [0]

        if sol["recur"]:
            time = str(recur_add + sol["day"] * 24 * 4 + sol["start_time"])
        else:
            # time = str(once_off_add[sol["day"]]+sol["start_time"])
            time = str(sol["start_time"])

        if "rooms_index" in sol.keys():
            building_id = []
            for room in sol["rooms_index"]:
                if small:
                    building_id += [str(self.small_rooms[room])]
                else:
                    building_id += [str(self.large_rooms[room])]
        else:
            building_id = [str(i) for i in sol["room_used"]]

        num_rooms = str(len(building_id))
        build_ids = " ".join(building_id)

        line = (
            start_line + str(sol["id"]) + " " + time + " " + num_rooms + " " + build_ids
        )

        return line

    def activity_string_to_sol(self, row_str):

        sol = dict()
        sol["keep"] = True
        if row_str[0] == "r":
            sol["recur"] = True
        else:
            sol["recur"] = False

        sol["id"] = int(row_str[1])

        recur_add = self.monday_index

        if sol["recur"]:
            time = int(row_str[2])
            day = int(time - recur_add) // (24 * 4)
            start_time = (int(time) - recur_add) % (24 * 4)
            sol["day"] = day
            sol["start_time"] = start_time
        else:
            sol["start_time"] = int(row_str[2])
            sol["day"] = (int(row_str[2]) + 11 * 4) // (24 * 4)
            time = str(sol["start_time"])

        room_index = row_str[2]

        if sol["recur"]:
            if self.recur[sol["id"]]["room_size"] == "S":
                small = True
            else:
                small = False
        else:
            if self.once_off[sol["id"]]["room_size"] == "S":
                small = True
            else:
                small = False

        # I think this part is not possible (or at least not straightforward to know)

        sol["room_used"] = [int(i) for i in row_str[4:]]

        return sol, small

    def assign_rooms(self, new_sol):
        build_id_to_real = [0, 1, 3, 4, 5, 6]
        build_id_to_ind = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}

        if new_sol["recur"]:
            info = self.recur[new_sol["id"]]

        else:
            info = self.once_off[new_sol["id"]]
        small = info["room_size"] == "S"

        avail_rooms = list()
        rooms_used = list()
        for building in self.build:
            rooms_used.append(0)
            if small:
                avail_rooms.append(building["num_small"])
            else:
                avail_rooms.append(building["num_big"])

        if small:
            room_sols = self.sol_small
        else:
            room_sols = self.sol_large
        if new_sol["recur"]:
            new_start_time = (
                self.monday_index + new_sol["day"] * 24 * 4 + new_sol["start_time"]
            )
        else:
            new_start_time = new_sol["start_time"]
        duration = info["duration"]

        times = list()
        if new_sol["recur"]:
            for i in range(4):
                for j in range(new_start_time, new_start_time + duration):
                    times += [j + i * 24 * 4 * 7]
        else:
            for j in range(new_start_time, new_start_time + duration):
                times += [j]

        for time in times:
            room_count = [0] * len(avail_rooms)
            for sol in room_sols:
                if new_sol["recur"] and sol["recur"] and sol["day"] != new_sol["day"]:
                    continue
                if sol["recur"]:
                    sol_duration = self.recur[sol["id"]]["duration"]
                    check_start_times = list()
                    start_time = (
                        self.monday_index + sol["day"] * 24 * 4 + sol["start_time"]
                    )
                    for i in range(4):

                        check_start_times += [start_time + i * 7 * 24 * 4]
                else:
                    check_start_times = [sol["start_time"]]
                    sol_duration = self.once_off[sol["id"]]["duration"]

                for start_time in check_start_times:
                    end_time = start_time + sol_duration

                    if start_time <= time < end_time:
                        for build_id in sol["room_used"]:
                            room_count[build_id_to_ind[build_id]] += 1
            for build_id, _ in enumerate(rooms_used):
                used = max(rooms_used[build_id], room_count[build_id])
                if used > avail_rooms[build_id]:
                    return False
                rooms_used[build_id] = used

        if sum(rooms_used) + info["num_rooms"] > sum(avail_rooms):
            return False

        new_sol["room_used"] = list()

        for build_id, used in enumerate(rooms_used):

            avail = avail_rooms[build_id] - used
            for _ in range(avail):
                new_sol["room_used"].append(build_id_to_real[build_id])
                if len(new_sol["room_used"]) == info["num_rooms"]:
                    return True
        return False

    def get_prices(self):
        prices = list()
        with open(load_price) as csvfile:
            spamreader = csv.reader(csvfile)

            for row in spamreader:
                prices.append(float(row[1]))
        return prices

    def get_base_load(self):
        base_load = [0 for _ in self.datetimes]
        """
        with open(base_load_profile) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                for i, _ in enumerate(base_load):
                    base_load[i] += float(row[i + 1])
                    #if row[0].startswith("Solar"):
                    #    base_load[i] += float(row[i+1])
                    #elif row[0].startswith("Building"):
                    #    base_load[i] += float(row[i+1])
        """
        with open(load_price) as csvfile:
            spamreader = csv.reader(csvfile)
            for i, row in enumerate(spamreader):
                base_load[i] = float(row[0])
        return base_load

    def get_objective_function(self):
        solutions = self.sol_small + self.sol_large
        solutions = [i for i in solutions if i["keep"]]
        final_load = copy.deepcopy(self.base_load)

        for sol in solutions:
            if not sol["recur"]:
                info = self.once_off[sol["id"]]
                start_times = [sol["start_time"]]
            else:
                info = self.recur[sol["id"]]
                start_times = list()

                start_time = self.monday_index + sol["day"] * 24 * 4 + sol["start_time"]
                week = 0
                while start_time + info["duration"] < len(self.datetimes) and week < 4:
                    start_times += [start_time]
                    start_time += 7 * 4 * 24
                    week += 1

            duration = info["duration"]
            power_per_time = info["num_rooms"] * info["load"]

            for start_time in start_times:
                for i in range(duration):

                    time = start_time + i
                    final_load[time] += power_per_time

        max_peak_cost = max(final_load) ** 2 * 0.005

        load_cost = sum(
            [self.prices[i] * value * 0.25 / 1000 for i, value in enumerate(final_load)]
        )

        once_off_bonus = 0
        for sol in solutions:
            if not sol["recur"]:
                info = self.once_off[sol["id"]]
                start_time = sol["start_time"]
                duration = info["duration"]

                start_dt = self.datetimes[start_time]
                end_dt = start_dt + duration * datetime.timedelta(minutes=15)
                if (
                    start_dt.weekday() < 5
                    and 9 <= start_dt.hour <= 16
                    and end_dt.weekday() < 5
                    and (
                        9 <= end_dt.hour <= 16
                        or (end_dt.hour == 17 and end_dt.minute == 0)
                    )
                ):
                    once_off_bonus += info["value"]
                else:
                    once_off_bonus += info["value"] - info["penalty"]
        # ("load_cost")
        # print(load_cost)
        # print("max_peak_cost")
        # print(max_peak_cost)
        tot_cost = load_cost + max_peak_cost - once_off_bonus
        return tot_cost

    def parse_solution_file(self, solution_file):
        self.sol_small = list()
        self.sol_large = list()
        with open(solution_file, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=" ")
            for row in csvreader:
                if row[0] == "r" or row[0] == "a":
                    sol, small = self.activity_string_to_sol(row)
                    # feasible = self.assign_rooms(sol)
                    if small:
                        self.sol_small.append(sol)
                    else:
                        self.sol_large.append(sol)
                elif row[0] == "c":
                    sol_batt = {"id": row[1], "time": row[2], "value": row[3]}
                    self.sol_batt.append(sol_batt)

    def plot_activities(self, save_fig=False):
        solutions = self.sol_small + self.sol_large
        visualise_activities(
            solutions,
            self.base_load,
            self.prices,
            self.recur,
            self.once_off,
            self.monday_index,
            self.sol_batt,
            self.batt,
            save_fig,
        )


def calculate_recur_cost(problem: Problem, once_off_sol):
    recur_info = problem.recur[once_off_sol["id"]]
    start_time = once_off_sol["start_time"]
    duration = recur_info["duration"]
    power_per_time = recur_info["num_rooms"] * recur_info["load"] * 0.25 / 1000
    total_price = 0

    for i in range(duration):
        time = start_time + i
        price = problem.prices[time]
        total_price += price * power_per_time

    return total_price


def calculate_once_off_cost(problem: Problem, once_off_sol):
    once_off_info = problem.once_off[once_off_sol["id"]]
    start_time = once_off_sol["start_time"]
    duration = once_off_info["duration"]
    power_per_time = once_off_info["num_rooms"] * once_off_info["load"] * 0.25 / 1000

    start_dt = problem.datetimes[start_time]
    end_dt = start_dt + duration * datetime.timedelta(minutes=15)
    if (
        start_dt.weekday() < 5
        and 9 <= start_dt.hour <= 16
        and end_dt.weekday() < 5
        and (9 <= end_dt.hour <= 16 or (end_dt.hour == 17 and end_dt.minute == 0))
    ):
        penalty = 0
    else:
        penalty = once_off_info["penalty"]
    bonus = once_off_info["value"]
    total_price = penalty - bonus
    for i in range(duration):
        time = start_time + i
        price = problem.prices[time]
        total_price += price * power_per_time

    return total_price


def count_keep(sols):
    count = 0
    for sol in sols:
        if not sol["recur"]:
            if sol["keep"]:
                count += 1
    return count


def create_tree_problem(tree_objects):
    node_list = [None] * len(tree_objects)
    for node in tree_objects:
        if node["num_preccedences"] == 0:
            nodes_below = []
            for node_j in tree_objects:
                if node["id"] in node_j["precedences"]:
                    nodes_below += [node_j]
            levels_above = 0
            levels_after = recur_levels_below(
                node_list, nodes_below, levels_above, tree_objects
            )
            node_list[node["id"]] = {
                "id": node["id"],
                "levels_above": levels_above,
                "levels_after": levels_after,
                "nodes_below": nodes_below,
                "precedences": node["precedences"],
            }

    return node_list


def recur_levels_below(node_list, old_nodes_below, old_levels_above, tree_objects):

    if old_nodes_below == []:
        return 0
    else:
        max_levels_after = 1
        for node in old_nodes_below:
            if node_list[node["id"]] is None:

                nodes_below = list()
                for node_j in tree_objects:
                    if node["id"] in node_j["precedences"]:
                        nodes_below += [node_j]

                levels_above = old_levels_above + 1
                levels_after = recur_levels_below(
                    node_list, nodes_below, levels_above, tree_objects
                )
                node_list[node["id"]] = {
                    "id": node["id"],
                    "levels_above": levels_above,
                    "levels_after": levels_after,
                    "nodes_below": nodes_below,
                    "precedences": node["precedences"],
                }
            else:
                levels_after = node_list[node["id"]]["levels_after"]
                if node_list[node["id"]]["levels_above"] <= old_levels_above:
                    node_list[node["id"]]["levels_above"] = old_levels_above + 1

                    # Recheck that with the update, the nodes below are still correct
                    nodes_below = list()
                    for node_j in tree_objects:
                        if node["id"] in node_j["precedences"]:
                            nodes_below += [node_j]
                    recur_levels_below(
                        node_list,
                        nodes_below,
                        node_list[node["id"]]["levels_above"],
                        tree_objects,
                    )

            if levels_after >= max_levels_after:
                max_levels_after = levels_after + 1

    return max_levels_after


def get_all_precedence_index(parent_id, tree, precedence_list):

    if parent_id in precedence_list:
        return

    precedence_list += [parent_id]
    tree_o_o = tree[parent_id]

    if tree_o_o["levels_above"] == 0:
        return
    else:

        for parent in tree_o_o["precedences"]:
            if parent not in precedence_list:
                get_all_precedence_index(parent, tree, precedence_list)
        return


def set_possible_days(node_list, tot_num_days=5):
    for item in node_list:
        lvls_abv = item["levels_above"]
        lvls_aftr = item["levels_after"]

        end_range = tot_num_days - lvls_aftr
        days_allowed = [i for i in range(lvls_abv, end_range)]
        item["days_allowed"] = days_allowed


def get_all_days(datetimes):
    current_week_day = -1
    all_days = list()
    for dt in datetimes:
        if current_week_day != dt.weekday():
            current_week_day = dt.weekday()
            all_days.append(current_week_day)

    return all_days


def parse_ppoi(row):
    ppoi = dict()
    ppoi["num_build"] = row[1]
    ppoi["num_solar"] = row[2]
    ppoi["num_batt"] = row[3]
    ppoi["num_recur"] = row[4]
    ppoi["num_once_off"] = row[5]
    return ppoi


def parse_build(row):
    build = dict()
    build["id"] = row[1]
    build["num_small"] = row[2]
    build["num_big"] = row[3]
    return build


def parse_solar(row):
    solar = dict()
    solar["id"] = row[1]
    solar["build_id"] = row[2]
    return solar


def parse_batt(row):
    batt = dict()
    batt["id"] = row[1]
    batt["build id"] = row[2]
    batt["capacity"] = row[3]
    batt["max_power"] = row[4]
    batt["efficiency"] = row[5]
    return batt


def parse_recur(row):
    recur = dict()
    recur["id"] = row[1]
    recur["num_rooms"] = row[2]
    recur["room_size"] = row[3]
    recur["load"] = row[4]
    recur["duration"] = row[5]
    recur["num_preccedences"] = row[6]
    recur["precedences"] = row[7:]
    return recur


def parse_once_off(row):
    once_off = dict()
    once_off["id"] = row[1]
    once_off["num_rooms"] = row[2]
    once_off["room_size"] = row[3]
    once_off["load"] = row[4]
    once_off["duration"] = row[5]
    once_off["value"] = row[6]
    once_off["penalty"] = row[7]
    once_off["num_preccedences"] = row[8]
    once_off["precedences"] = row[9:]
    return once_off


def perdelta(start, end, delta):
    output = list()
    curr = start
    while curr < end:
        output.append(curr)
        curr += delta

    return output


def first_monday_index(datetimes):

    for i, datetime in enumerate(datetimes):
        if datetime.weekday() == 0:
            return i


def int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


if __name__ == "__main__":
    inst_path = "instances_p2/phase2_instance_large_0.txt"
    # inst_path = "instances_p2/phase2_instance_small_1.txt"
    # inst_path = "instances_p2/phase2_instance_small_4.txt"
    # inst_path = "instances_p2/phase2_instance_large_4.txt"
    start = datetime.datetime(2020, 11, 1)
    end = datetime.datetime(2020, 12, 1)
    prob = Problem(inst_path, start, end)
    sol_path = "solutions_sched_p2/sol_small_0_31035.608755113244.txt"
    sol_path = "solutions_impr_p2/sol_small_1_27511.729339366688.txt"
    sol_path = "final_submission/sol_large_0_26105.963835665723.txt"
    # sol_path = "solutions_sched_p2/sol_small_1_28286.66648571543.txt"
    # sol_path = "solutions_sched_p2/sol_small_1_29536.041886727216.txt"
    # sol_path = "solutions_sched_p2/sol_small_4_29380.16464723889.txt"
    # sol_path = "solutions_sched_p2/sol_large_3_27209.306987244774.txt"
    # sol_path = "sample_submission_p2/phase2_instance_solution_large_3.txt"
    prob.parse_solution_file(sol_path)
    prob.plot_activities()
