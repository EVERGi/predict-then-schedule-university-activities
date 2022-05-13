from instance_parser import Problem, perdelta, calculate_once_off_cost
import datetime
import random
import copy
import subprocess


def random_solution_days(problem: Problem):

    tree_once_off = problem.tree_once_off
    tree_recur = problem.tree_recur

    sol_once_off = [random.choice(i["days_allowed"]) for i in tree_once_off]
    sol_recur = [random.choice(i["days_allowed"]) for i in tree_recur]
    # print(sol_once_off)
    sol_recur = sol_tree_push(tree_recur, sol_recur, 5)

    # print(sol_once_off)
    # sol_once_off = sol_tree_push(tree_once_off, sol_once_off, problem.num_weekdays)
    sol_once_off = sol_tree_push(tree_once_off, sol_once_off, problem.num_all_days)
    # print(sol_once_off)
    return sol_recur, sol_once_off


def random_solution_times(problem: Problem):
    possible_start = [9 * 4 + i for i in range(17 * 4 - 9 * 4 + 1)]

    sol_time_recur = [
        random.choice(possible_start[: -i["duration"]]) for i in problem.recur
    ]
    sol_time_once_off = [
        random.choice(possible_start[: -i["duration"]]) for i in problem.once_off
    ]

    return sol_time_recur, sol_time_once_off


def random_solution_battery(problem: Problem):
    num_batteries = len(problem.batt)
    batt_sched = [
        [random.randint(0, 3) for _ in problem.datetimes] for _ in range(num_batteries)
    ]
    return batt_sched


def create_feasible_schedule_batt(problem: Problem, batt_sched):
    num_batteries = len(problem.batt)

    cap = [problem.batt[i]["capacity"] for i in range(num_batteries)]
    for id, sched in enumerate(batt_sched):
        for time, value in enumerate(sched):
            max_cap = problem.batt[id]["capacity"]
            max_charge = problem.batt[id]["max_power"]
            if value == 0:
                next_cap = cap[id] + max_charge
                if next_cap <= max_cap:
                    problem.sol_batt.append({"id": id, "time": time, "value": 0})
                    cap[id] = next_cap
            elif value == 2:
                next_cap = cap[id] - max_charge
                if next_cap >= 0:
                    problem.sol_batt.append({"id": id, "time": time, "value": 2})
                    cap[id] = next_cap


def sol_tree_push(tree_recur, sol_recur, max_levels):
    new_sol_recur = copy.copy(sol_recur)
    for level in range(0, max_levels):
        for index, day in enumerate(new_sol_recur):
            if level == tree_recur[index]["levels_above"] == 0:
                new_sol_recur[index] = day
            elif tree_recur[index]["levels_above"] == level:
                new_day = day
                for precedent in tree_recur[index]["precedences"]:
                    if new_day <= new_sol_recur[precedent]:
                        new_day = new_sol_recur[precedent] + 1
                new_sol_recur[index] = new_day
    return new_sol_recur


def create_feasible_schedule(
    problem: Problem, sol_recur, sol_once_off, time_recur=None, time_once_off=None
):
    # slots_per_day = 4*(17-9)

    recur = problem.recur
    once_off = problem.once_off

    small_rooms = problem.small_rooms
    large_rooms = problem.large_rooms
    small_slots = [
        [[[9 * 4, 17 * 4]] for _ in range(len(small_rooms))] for _ in range(5)
    ]
    # small_slots = [[[[9*4, 17*4]]]*len(small_rooms)]*5

    large_slots = [
        [[[9 * 4, 17 * 4]] for _ in range(len(large_rooms))] for _ in range(5)
    ]
    # print(small_slots)
    # print(large_slots)

    for day_to_schedule in range(5):
        activities_to_schedule = list()
        times_to_schedule = list()
        for id, day_of_week in enumerate(sol_recur):
            if day_of_week == day_to_schedule:
                activities_to_schedule.append(recur[id])

                if time_recur is not None:
                    times_to_schedule.append(time_recur[id])
                else:
                    times_to_schedule.append(None)

        (
            small_slots[day_to_schedule],
            large_slots[day_to_schedule],
            sol_small,
            sol_large,
        ) = fill_in_schedule(
            small_slots[day_to_schedule],
            large_slots[day_to_schedule],
            activities_to_schedule,
            day_to_schedule,
            True,
            times_to_schedule,
        )
        problem.sol_small += sol_small
        problem.sol_large += sol_large

    small_slots = week_slot_to_month(small_slots, problem)
    large_slots = week_slot_to_month(large_slots, problem)

    new_small_slots = copy.copy(small_slots)
    new_large_slots = copy.copy(large_slots)

    times_to_schedule = [
        time_once_off[id] + day_of_week * 24 * 4 - 11 * 4
        for id, day_of_week in enumerate(sol_once_off)
    ]
    times_to_schedule = list()
    activities_to_schedule = list()
    for id, day_of_week in enumerate(sol_once_off):
        time = time_once_off[id] + day_of_week * 24 * 4 - 11 * 4
        if time < len(problem.datetimes):
            times_to_schedule.append(time)
            activities_to_schedule.append(once_off[id])

    new_small_slots[0], new_large_slots[0], sol_small, sol_large = fill_in_schedule(
        small_slots[0],
        large_slots[0],
        activities_to_schedule,
        0,
        False,
        times_to_schedule,
    )

    problem.sol_small += sol_small
    problem.sol_large += sol_large

    # for sol in problem.sol_small+problem.sol_large:
    #    if not sol["recur"]:
    #        sol["start_time"] -= 11*4
    #        sol["end_time"] -= 11*4


def fill_in_schedule(
    small_slots, large_slots, activities, day, recur, times_to_schedule
):
    # small_slots = copy.deepcopy(small_slots)
    # large_slots = copy.deepcopy(large_slots)
    unsorted_activities_small = list()
    unsorted_activities_large = list()
    for ind, activity in enumerate(activities):
        new_act_to_sort = {
            "id": activity["id"],
            "num_rooms": activity["num_rooms"],
            "duration": activity["duration"],
            "time": times_to_schedule[ind],
        }
        new_act_to_sort["size"] = (
            new_act_to_sort["num_rooms"] * new_act_to_sort["duration"]
        )

        if new_act_to_sort["time"] > len(small_slots):
            pass

        if activity["room_size"] == "S":
            unsorted_activities_small.append(new_act_to_sort)
        else:
            unsorted_activities_large.append(new_act_to_sort)

    sorted_activities_small = sorted(
        unsorted_activities_small, key=lambda k: k["size"]
    )[::-1]
    sorted_activities_large = sorted(
        unsorted_activities_large, key=lambda k: k["size"]
    )[::-1]

    act_sol_small = []
    for activity in sorted_activities_small:
        # if activity["time"] == None:
        #    small_slots, sol = fit_one_activity(small_slots, activity, day, recur)
        # else:
        small_slots, sol = fit_one_activity_with_time(small_slots, activity, day, recur)
        act_sol_small.append(sol)

    act_sol_large = []
    for activity in sorted_activities_large:
        # if activity["time"] == None:
        #    large_slots, sol = fit_one_activity(large_slots, activity, day, recur)
        # else:
        large_slots, sol = fit_one_activity_with_time(large_slots, activity, day, recur)
        act_sol_large.append(sol)

    # if solution_optional == []:
    #    return True
    return small_slots, large_slots, act_sol_small, act_sol_large


def fit_one_activity_with_time(schedule, activity, day, recur):

    possible_to_schedule = False

    time_start = list()

    count = 0
    for i in range(2 * (17 * 4 - 9 * 4)):
        time_start.append(activity["time"] + count)
        if i % 2 == 0:
            count = -count + 1
        else:
            count = -count

    for act_start in time_start:
        if not recur and act_start // (24 * 4) != activity["time"] // (24 * 4):
            continue

        act_end = act_start + activity["duration"]
        rooms_to_store = list()
        room_avail_ind = list()
        for ind, room in enumerate(schedule):
            for avail_ind, avail in enumerate(room):
                if avail[0] <= act_start < avail[1] and act_end <= avail[1]:
                    rooms_to_store.append(ind)
                    room_avail_ind.append(avail_ind)
                    break
            if len(rooms_to_store) == activity["num_rooms"]:

                possible_to_schedule = True
                break
        if possible_to_schedule:
            break

    if not possible_to_schedule:
        return False

    for ind, rooms in enumerate(rooms_to_store):
        avail_ind = room_avail_ind[ind]
        start = schedule[rooms][avail_ind][0]
        end = schedule[rooms][avail_ind][1]
        act_end = act_start + activity["duration"]
        if start != act_start and end != act_end:
            schedule[rooms][avail_ind : avail_ind + 1] = [
                [start, act_start],
                [act_end, end],
            ]
        elif start == act_start and end == act_end:
            del schedule[rooms][avail_ind]
        elif start == act_start:
            schedule[rooms][avail_ind] = [act_end, end]
        elif end == act_end:
            schedule[rooms][avail_ind] = [start, act_start]
    # print(schedule)
    act_sol = {
        "id": activity["id"],
        "start_time": act_start,
        "end_time": act_end,
        "rooms_index": rooms_to_store,
        "day": day,
        "recur": recur,
        "keep": True,
    }
    return schedule, act_sol


"""
def week_slot_to_month_legacy(slots, problem):
    week_slots = list()
    for day in problem.weekdays:
        week_slots.append(copy.deepcopy(slots[day]))
    return week_slots
"""


def week_slot_to_month(slots, problem):
    month_slots = [[[[0, 0]] for _ in slots[0]]]
    monday_encountered = False

    for month_day, day in enumerate(problem.all_days):

        if day == 0:
            monday_encountered = True

        if day >= 5 or not monday_encountered:
            for room, _ in enumerate(slots[0]):
                # If it's week-end or so, add the whole day to the schedule
                if month_day == 0:
                    month_slots[0][room][-1][1] += (24 - 11) * 4
                elif month_day == len(problem.all_days) - 1:
                    month_slots[0][room][-1][1] += 11 * 4
                else:
                    month_slots[0][room][-1][1] += 24 * 4
        else:

            for room, sched_per_day in enumerate(slots[day]):
                end_last_day = month_slots[0][room][-1][1]

                # Fill in all the timeslots
                first_encountered = True
                for time_range in sched_per_day:
                    if time_range[0] == 9 * 4:
                        first_encountered = False
                        if month_day == len(problem.all_days) - 1:
                            if time_range[1] <= 11 * 4:
                                month_slots[0][room][-1][1] = (
                                    end_last_day + time_range[1]
                                )
                            else:
                                month_slots[0][room][-1][1] = end_last_day + 11 * 4

                        else:
                            month_slots[0][room][-1][1] = end_last_day + time_range[1]
                    else:
                        if first_encountered:
                            month_slots[0][room][-1][1] = end_last_day + 9 * 4
                        first_encountered = False
                        if month_day == len(problem.all_days) - 1:
                            if time_range[0] < 11 * 4:
                                if time_range[1] <= 11 * 4:
                                    new_slot = [
                                        end_last_day + time_range[0],
                                        end_last_day + time_range[1],
                                    ]
                                    month_slots[0][room].append(new_slot)
                                else:
                                    new_slot = [
                                        end_last_day + time_range[0],
                                        end_last_day + 11 * 4,
                                    ]
                                    month_slots[0][room].append(new_slot)
                            else:
                                pass
                        else:
                            new_slot = [
                                end_last_day + time_range[0],
                                end_last_day + time_range[1],
                            ]
                            month_slots[0][room].append(new_slot)
                if month_day == len(problem.all_days) - 1:
                    continue
                elif month_slots[0][room][-1][1] == end_last_day + 17 * 4:
                    month_slots[0][room][-1][1] = end_last_day + 24 * 4
                else:
                    new_slot = [end_last_day + 17 * 4, end_last_day + 24 * 4]
                    month_slots[0][room].append(new_slot)

    return month_slots


def remove_once_off(problem: Problem):
    # print(problem.sol_small)
    # print(problem.sol_large)
    once_off_sols = [i for i in problem.sol_small if not i["recur"]] + [
        i for i in problem.sol_large if not i["recur"]
    ]
    once_off_sols = sorted(once_off_sols, key=lambda k: k["id"])

    list_value = [None for _ in once_off_sols]
    list_indiv_value = [None for _ in once_off_sols]
    min_val = 1000000
    min_id = -1

    sel_sols = {once_off["id"]: i for i, once_off in enumerate(once_off_sols)}

    sol_sels = {i: once_off["id"] for i, once_off in enumerate(once_off_sols)}

    for once_off in once_off_sols:
        id = once_off["id"]
        list_indiv_value[sel_sols[id]] = calculate_once_off_cost(problem, once_off)

    for once_off in once_off_sols:
        id = once_off["id"]
        list_id = sel_sols[id]
        all_prec = problem.tree_once_off[id]["all_precedences"]
        tot_val = sum([list_indiv_value[sel_sols[i]] for i in all_prec])
        if list_value[list_id] is None:
            list_value[list_id] = tot_val
        if tot_val < min_val and list_indiv_value[list_id] < 0:
            min_val = tot_val
            min_id = id
    while min_val < 0 and min_id != -1:
        all_prec = problem.tree_once_off[min_id]["all_precedences"]
        list_value = [
            0 if sol_sels[i] in all_prec or zer_val == 0 else None
            for i, zer_val in enumerate(list_value)
        ]
        list_indiv_value = [
            0 if sol_sels[i] in all_prec or list_value[i] == 0 else old_val
            for i, old_val in enumerate(list_indiv_value)
        ]

        min_val = 1000000
        min_id = -1
        for once_off in once_off_sols:
            id = once_off["id"]
            list_id = sel_sols[id]
            all_prec = problem.tree_once_off[id]["all_precedences"]
            tot_val = sum([list_indiv_value[sel_sols[i]] for i in all_prec])
            if list_value[list_id] is None:
                list_value[list_id] = tot_val
            if tot_val < min_val and list_indiv_value[list_id] < 0:
                min_val = tot_val
                min_id = id

    for sol in once_off_sols:
        if list_value[sel_sols[sol["id"]]] == 0:
            sol["keep"] = True
        else:
            sol["keep"] = False


def get_objective_function(instance_path, solution_path):
    jar_file = "Optim_eval/target/evaluate_instance.jar"
    a = subprocess.check_output(
        ["java", "-jar", jar_file, instance_path, solution_path]
    ).decode("utf-8")
    # print(a)
    b = a.split("\n")
    res = float(b[-2])

    return res


if __name__ == "__main__":
    inst_path = "instances_p2/phase2_instance_small_1.txt"
    # inst_path = "instances_p2/phase2_instance_small_1.txt"
    # inst_path = "instances_p2/phase2_instance_small_4.txt"
    # inst_path = "instances_p2/phase2_instance_large_4.txt"
    start = datetime.datetime(2020, 11, 1)
    end = datetime.datetime(2020, 12, 1)
    prob = Problem(inst_path, start, end)
    sol_path = "solutions_sched_p2/sol_small_0_31035.608755113244.txt"
    sol_path = "solutions_impr_p2/sol_small_1_27511.729339366688.txt"
    # sol_path = "solutions_sched_p2/sol_small_1_28286.66648571543.txt"
    # sol_path = "solutions_sched_p2/sol_small_1_29536.041886727216.txt"
    # sol_path = "solutions_sched_p2/sol_small_4_29380.16464723889.txt"
    # sol_path = "solutions_sched_p2/sol_large_3_27209.306987244774.txt"
    # sol_path = "sample_submission_p2/phase2_instance_solution_large_3.txt"
    print(get_objective_function(inst_path, sol_path))
