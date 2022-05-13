import matplotlib.pyplot as plt
import os
import csv
import statistics
import numpy as np
from visualise_sol import visualise_activities
import datetime
from instance_parser import Problem
from colour import Color

save_figs = False


def plot_generation_results(result_dir):

    pygmo_small = dict()
    pygmo_large = dict()

    pygad_small = dict()
    pygad_large = dict()

    pygmo_small_agr = dict()
    pygmo_large_agr = dict()

    pygad_small_agr = dict()
    pygad_large_agr = dict()

    index_iter = dict()
    max_score = dict()

    for filepath in os.listdir(result_dir):
        small = filepath.find("small") != -1
        pygmo = filepath.find("pygmo") != -1

        if small and pygmo:
            used_dict = pygmo_small
            used_agr = pygmo_small_agr
        elif not small and pygmo:
            used_dict = pygmo_large
            used_agr = pygmo_large_agr
        elif small and not pygmo:
            used_dict = pygad_small
            used_agr = pygad_small_agr
        else:
            used_dict = pygad_large
            used_agr = pygad_large_agr

        for key in index_iter.keys():
            index_iter[key] = 0

        for key in max_score.keys():
            max_score[key] = 200000
        with open(result_dir + filepath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == "population":
                    continue

                pop = int(row[0])
                if pop not in used_dict.keys():
                    used_dict[pop] = list()
                    used_agr[pop] = list()
                    index_iter[pop] = 0
                    max_score[pop] = 200000

                if pygmo:
                    score = float(row[2])
                    gen = int(row[1])
                else:
                    score = -float(row[2])
                    gen = int(row[1]) - 1

                if len(used_agr[pop]) <= gen:
                    used_agr[pop].append(list())

                if max_score[pop] > score:
                    max_score[pop] = score

                used_agr[pop][gen].append(score)

                if len(used_dict[pop]) == index_iter[pop]:
                    used_dict[pop].append(max_score[pop])
                else:
                    used_dict[pop][index_iter[pop]] += max_score[pop]

                index_iter[pop] += 1

        for pop in used_dict.keys():
            prev_val = used_dict[pop][0]
            for ind, value in enumerate(used_dict[pop]):

                if prev_val - value > 15000:
                    used_dict[pop] = used_dict[pop][:ind]
                    break
                prev_val = value

    base_small = 140609.53
    base_large = 134799.65

    impr_small = 138115.58
    impr_large = 133046.43

    batt_small = 127880.19
    batt_large = 125811.76

    red = Color("red")
    if len(pygmo_small.keys()) != 0:
        colors_CMAES = list(red.range_to(Color("#FDDA0D"), len(pygmo_small.keys())))[
            ::-1
        ]
    blue = Color("blue")
    if len(pygad_small.keys()) != 0:
        colors_GA = list(blue.range_to(Color("#90EE90"), len(pygad_small.keys())))[::-1]

    sorted_keys = list(pygad_small.keys())
    sorted_keys.sort()

    plt.figure()
    ax = plt.axes()

    end_plot = 5e6
    for i, pop in enumerate(sorted(pygad_small.keys())):
        limit = int(end_plot / pop) + 1
        x_axis = [i * pop for i, _ in enumerate(pygad_small[pop])]
        ax.plot(
            x_axis[:limit],
            pygad_small[pop][:limit],
            color=colors_GA[i].rgb,
            label="GA " + str(pop),
        )
    for i, pop in enumerate(sorted(pygmo_small.keys())):
        limit = int(end_plot / pop) + 1
        x_axis = [i * pop for i, _ in enumerate(pygmo_small[pop])]
        ax.plot(
            x_axis[:limit],
            pygmo_small[pop][:limit],
            color=colors_CMAES[i].rgb,
            label="CMA-ES " + str(pop),
        )
    ax.axhline(y=base_small, color="orange", linestyle="--", label="Base schedule")
    ax.axhline(y=impr_small, color="g", linestyle="--", label="Improved schedule")
    ax.axhline(y=batt_small, color="y", linestyle="--", label="Battery schedule")

    plt.xlabel("Number of iterations")
    plt.ylabel("Summed cost of the 5 small instances")

    # plt.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.85])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    # plt.tight_layout()

    # plt.ylim((1.3e5,1.6e5))

    if save_figs:
        plt.savefig("../Figures/small_base_long.png", bbox_inches="tight")

    plt.figure()
    ax = plt.axes()

    end_plot = 1.5e6
    for i, pop in enumerate(sorted(pygad_large.keys())):
        limit = int(end_plot / pop) + 1
        x_axis = [i * pop for i, _ in enumerate(pygad_large[pop])]
        ax.plot(
            x_axis[:limit],
            pygad_large[pop][:limit],
            color=colors_GA[i].rgb,
            label="GA pop " + str(pop),
        )
    for i, pop in enumerate(sorted(pygmo_large.keys())):
        limit = int(end_plot / pop) + 1
        x_axis = [i * pop for i, _ in enumerate(pygmo_large[pop])]
        ax.plot(
            x_axis[:limit],
            pygmo_large[pop][:limit],
            color=colors_CMAES[i].rgb,
            label="CMA-ES pop " + str(pop),
        )
    ax.axhline(y=base_large, color="orange", linestyle="--", label="Base schedule")
    ax.axhline(y=impr_large, color="g", linestyle="--", label="Improved schedule")
    ax.axhline(y=batt_large, color="y", linestyle="--", label="Battery schedule")

    plt.xlabel("Number of iterations")
    plt.ylabel("Summed best cost of the 5 large instances")

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # plt.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.85])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    # ax.get_legend().remove()
    # plt.tight_layout()

    # plt.ylim((1.25e5,1.55e5))

    if save_figs:
        plt.savefig(
            "../Figures/large_base_long.png",
            bbox_inches="tight",
        )

    """
    pygad_small_mean = np.array([statistics.mean(i) for i in pygad_small_agr[100]])
    pygad_large_mean = np.array([statistics.mean(i) for i in pygad_large_agr[100]])
    pygmo_small_mean = np.array([statistics.mean(i) for i in pygmo_small_agr[100]])
    pygmo_large_mean = np.array([statistics.mean(i) for i in pygmo_large_agr[100]])

    pygad_small_std = np.array([statistics.stdev(i) if len(i)>1 else 0.0 for i in pygad_small_agr[100]])
    pygad_large_std = np.array([statistics.stdev(i) if len(i)>1 else 0.0 for i in pygad_large_agr[100]])
    pygmo_small_std = np.array([statistics.stdev(i) if len(i)>1 else 0.0 for i in pygmo_small_agr[100]])
    pygmo_large_std = np.array([statistics.stdev(i) if len(i)>1 else 0.0 for i in pygmo_large_agr[100]])
    
    plt.figure()
    plt.plot(pygad_small_mean, label = "GA small")
    plt.plot(pygad_small_mean+pygad_small_std, linestyle='--', label = "GA small")
    plt.plot(pygad_small_mean-pygad_small_std, linestyle='--', label = "GA small")
    plt.axhline(y=base_large, color='r', linestyle='--', label="Base solution")
    plt.axhline(y=impr_large, color='g', linestyle='--', label="Improved solution")
    plt.axhline(y=batt_large, color='y', linestyle='--', label="Battery solution")
    """


def plot_improvement_and_battery(impr_file, battery_file):
    with open(impr_file) as csvfile:
        reader = csv.reader(csvfile)
        small_v1 = list()
        small_v2 = list()

        large_v1 = list()
        large_v2 = list()

        for row in reader:
            if row[0] == "instance":
                continue
            if row[0].startswith("small") and row[1] == "1":
                small_v1.append([float(row[2]), float(row[3])])
            elif row[0].startswith("small") and row[1] == "2":
                small_v2.append([float(row[2]), float(row[3])])
            elif row[0].startswith("large") and row[1] == "1":
                large_v1.append([float(row[2]), float(row[3])])
            else:
                large_v2.append([float(row[2]), float(row[3])])

    impr_box_plot = (
        [[i[0] for i in small_v1]]
        + [[i[1] for i in small_v1]]
        + [[i[0] for i in small_v2]]
        + [[i[1] for i in small_v2]]
        + [[i[0] for i in large_v1]]
        + [[i[1] for i in large_v1]]
        + [[i[0] for i in large_v2]]
        + [[i[1] for i in large_v2]]
    )

    with open(battery_file) as csvfile:
        small_batt = list()
        large_batt = list()

        reader = csv.reader(csvfile)

        for row in reader:
            if row[0] == "instance":
                continue
            if row[0].startswith("small"):
                small_batt.append(float(row[1]))
            else:
                large_batt.append(float(row[1]))

    batt_box_plot = [small_batt] + [large_batt]

    plt.figure()

    names = ["small \np1 v1", "small \np2 v1", "small \np1 v2", "small \np2 v2"]
    names += ["large \np1 v1", "large \np2 v1", "large \np1 v2", "large \np2 v2"]
    names += ["small \nbattery", "large \nbattery"]

    impr_box_plot = [val for i, val in enumerate(impr_box_plot) if i % 2 == 1]

    names = ["small\nkept", "small\nremoved"]
    names += ["large\nkept", "large\nremoved"]
    names += ["small\nbattery", "large\nbattery"]

    plt.axvline(4.5, color="black", linestyle="--")  # , linewidth=1)

    plt.boxplot(impr_box_plot + batt_box_plot)

    plt.xticks([i + 1 for i in range(len(names))], names)
    plt.ylabel("Cost improvement for 1 instance")
    plt.tight_layout()
    if save_figs:
        plt.savefig("../Figures/improvement_comparison.png")


def plot_all_results():
    plot_generation_results("generation_results/")

    plot_improvement_and_battery("improvement_results.csv", "improvement_battery.csv")

    inst_path = "instances_p2/phase2_instance_small_0.txt"
    start = datetime.datetime(2020, 11, 1)
    end = datetime.datetime(2020, 12, 1)
    prob = Problem(inst_path, start, end)
    sol_path = "sol_small_0_27455.75466152492.txt"
    prob.parse_solution_file(sol_path)
    prob.plot_activities()


if __name__ == "__main__":
    plot_generation_results("generation_results/")

    plot_improvement_and_battery("improvement_results.csv", "improvement_battery.csv")

    inst_path = "instances_p2/phase2_instance_small_0.txt"
    start = datetime.datetime(2020, 11, 1)
    end = datetime.datetime(2020, 12, 1)
    prob = Problem(inst_path, start, end)
    sol_path = "battery_schedule/sol_small_0_27455.75466152492.txt"
    prob.parse_solution_file(sol_path)
    prob.plot_activities(True)

    plt.show()
