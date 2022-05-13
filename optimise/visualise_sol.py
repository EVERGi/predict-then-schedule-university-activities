import matplotlib.pyplot as plt
import copy
import matplotlib.cm as cm
import numpy as np
import random


def visualise_activities(
    solutions,
    base_load,
    prices,
    recur_info,
    once_off_info,
    monday_index,
    bat_sols,
    batt_info,
    save_fig=False,
):
    # print(base_load)
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    x_axis = [i for i, _ in enumerate(base_load)]

    final_load_to_plot = [copy.deepcopy(base_load)]
    for load_num, sol in enumerate(solutions):
        if not sol["recur"]:

            info = once_off_info[sol["id"]]
            start_times = [sol["start_time"]]
        else:
            info = recur_info[sol["id"]]
            start_times = list()

            start_time = monday_index + sol["day"] * 24 * 4 + sol["start_time"]
            week = 0
            while start_time + info["duration"] < len(base_load) and week < 4:
                start_times += [start_time]
                start_time += 7 * 4 * 24
                week += 1

        duration = info["duration"]
        power_per_time = info["num_rooms"] * info["load"]

        for start_time in start_times:
            for i in range(duration):

                time = start_time + i
                final_load_to_plot[-1][time] += power_per_time

        final_load_to_plot.append(copy.deepcopy(final_load_to_plot[-1]))

    # colors = random.shuffle([i for i in cm.rainbow(np.linspace(0, 1, len(solutions)+1))])
    colors = cm.rainbow(np.linspace(0, 1, len(solutions) + 1))
    colors = colors[::-1]
    colors[-1] = colors[0]
    # random.shuffle(colors)
    # print(colors)
    new_x_axis = list()
    for x in x_axis:
        new_x_axis += [x - 0.5, x + 0.5]
    # x_axis = [x-0.5,x+0.5 for x in x_axis]
    for color_id, final_load in enumerate(final_load_to_plot[::-1]):
        new_final_load = list()
        for value in final_load:
            new_final_load += [value, value]
        # markerline, stemlines, baseline = plt.stem(x_axis, final_load ,markerfmt=" ", basefmt=" ")
        # plt.setp(stemlines, 'color', colors[color_id])
        # if color_id != len(final_load_to_plot)-1:
        # if color_id+1 >= len(solutions) or solutions[::-1][color_id+1]["recur"]:
        if color_id - 1 == -1 or solutions[::-1][color_id - 1]["recur"]:
            ax1.fill_between(new_x_axis, new_final_load, color=colors[color_id])
        else:
            ax1.fill_between(new_x_axis, new_final_load, color="black")
        # else:
        #    plt.fill_between(new_x_axis, new_final_load)

    # new_base_load = list()
    # for value in base_load:
    #    new_base_load += [value, value]

    # plt.fill_between(new_x_axis, new_base_load)

    all_9am_lines = [
        monday_index + 9 * 4 + i * 24 * 4 - 0.5 for i in range(30) if i % 7 < 5
    ]

    all_17pm_lines = [
        monday_index + 17 * 4 + (i - 1) * 24 * 4 - 0.5
        for i in range(30)
        if 0 < i % 7 < 6
    ]
    for x in all_9am_lines + all_17pm_lines:
        ax1.axvline(x, color="orange", linestyle="--", linewidth=1)

    ax2.plot(prices, color="b")

    ax2.set_ylim(ax1.get_ylim()[0] / 2, ax1.get_ylim()[1] / 2)

    ax1.set_xlabel("Time (15 min steps)")
    ax1.set_ylabel("Power (kW)")
    ax2.set_ylabel("Electricity wholesale price ($/MWh)", color="b")

    plt.tight_layout()
    if save_fig:
        plt.savefig("../Figures/example_schedule.png")

    plt.figure()

    final_load = final_load_to_plot[-1]
    batt_load = copy.deepcopy(final_load)
    for batt_sol in bat_sols:
        id = int(batt_sol["id"])
        efficiency = batt_info[id]["efficiency"]
        # capacity = batt_info[id]["capacity"]
        max_power = batt_info[id]["max_power"]

        if batt_sol["value"] == "2":
            batt_load[int(batt_sol["time"])] -= (efficiency**0.5) * max_power
        if batt_sol["value"] == "0":
            batt_load[int(batt_sol["time"])] += 1 / (efficiency**0.5) * max_power

    new_final_load = list()
    for value in final_load:
        new_final_load += [value, value]

    new_batt_load = list()
    for value in batt_load:
        new_batt_load += [value, value]

    new_prices = list()
    for value in prices:
        new_prices += [value, value]

    plt.plot(new_x_axis, new_final_load)
    plt.plot(new_x_axis, new_batt_load)
    plt.plot(new_x_axis, new_prices)
    plt.tight_layout()
    if save_fig:
        plt.savefig("../Figures/example_battery.png")
