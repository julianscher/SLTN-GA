import os
import time

import dill
import numpy as np
from matplotlib import pyplot as plt

from src.experiments.base.evol_routines import er13, er7, er8, er9, er10, er6, er14, er15
from src.experiments.base.init_routines import ir1, ir11, ir5, ir6, ir7, ir8, ir9, ir12, ir14
from src.genetic_algorithm.helper_functions import get_leaves_dfs, get_genealogy_bfs


def fr1(population, tmp_log):
    duplicates = 0
    for occurrence in tmp_log["genomes_occurrences"].values():
        if occurrence > 1:
            duplicates += occurrence - 1

    print("duplicates:", duplicates)
    print("number_of_different_genomes:", len(tmp_log["genomes_occurrences"]))

    population.evaluate_by_accuracy()
    print(population.__str__())

    print("--- %s seconds ---" % (time.time() - tmp_log["start_time"]))


def fr2(population, tmp_log):
    plt.figure(figsize=(9, 6))
    plt.plot(tmp_log["loss_development"])
    plt.title("loss development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("loss of best individual", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/loss_development")
    else:
        plt.show()


def fr3(population, tmp_log):
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))
    accuracies_in_percent = [100*acc for acc in tmp_log["accuracy_development"]]
    plt.plot(accuracies_in_percent, linewidth=5)
    # plt.title("Accuracy Development", fontsize=20)
    plt.xlabel("Generation", fontsize=50)
    plt.ylabel("Training Accuracy (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.grid(visible=True, which='both', linestyle='-', axis='y')
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/accuracy_development")
    else:
        plt.show()


def fr4(population, tmp_log):
    # Check how often, after recombination the child was better than its parents throughout the genealogy
    better_both_count = 0
    better_one_count = 0
    worse = 0
    for ind in population.individuals:
        genealogy = get_genealogy_bfs(ind, population.age)
        for individual in genealogy.keys():
            if len(genealogy.get(individual)) == 2:
                if individual.fitness > genealogy.get(individual)[0].fitness and \
                        individual.fitness > genealogy.get(individual)[1].fitness:
                    better_both_count += 1
                elif individual.fitness > genealogy.get(individual)[0].fitness or \
                        individual.fitness > genealogy.get(individual)[1].fitness:
                    better_one_count += 1
                else:
                    worse += 1
    print("better_both_count:", better_both_count)
    print("better_one_count:", better_one_count)
    print("worse:", worse)


def fr5(population, tmp_log):
    for individual in population.individuals:
        leaves = get_leaves_dfs(individual, population.age)
        leaves_that_are_in_the_final_population_count = 0
        for leaf in leaves:
            if leaf in population.individuals:
                leaves_that_are_in_the_final_population_count += 1
            print(leaf.origin, leaf.age)
        print(leaves_that_are_in_the_final_population_count)


def fr6(population, tmp_log):
    better_both_counts = [0] + [a for (a, b, c) in tmp_log["goodness_of_children"]]
    better_one_counts = [0] + [b for (a, b, c) in tmp_log["goodness_of_children"]]
    worse_counts = [0] + [c for (a, b, c) in tmp_log["goodness_of_children"]]
    generations = list(range(len(better_both_counts)))
    plt.figure(figsize=(9, 6))
    plt.bar(x=generations, height=worse_counts, width=0.8, align="edge", label="worse_count")
    plt.bar(x=generations, height=better_one_counts, width=0.8, bottom=np.array(worse_counts),
           align="edge", label="better_one_count")
    plt.bar(x=generations, height=better_both_counts, width=0.8,
           bottom=np.array(worse_counts)+np.array(better_one_counts), align="edge", label="better_both_counts")
    plt.xlabel("generation")
    plt.ylabel("children")
    plt.title("Recombine child-parents-goodness comparison")
    plt.legend()
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/recombine_child_parents_goodness_comparison")
    else:
        plt.show()


def fr7(population, tmp_log):
    top_recombine_counts = [0] + tmp_log["top_recombine_counts"]
    top_mutate_counts = [0] + tmp_log["top_mutate_counts"]
    top_migrate_counts = [0] + tmp_log["top_migrate_counts"]
    generations = list(range(len(top_recombine_counts)))
    plt.figure(figsize=(9, 6))

    # stack plot
    plt.stackplot(generations, top_migrate_counts, top_mutate_counts, top_recombine_counts,
                  labels=["migration", "mutation", "recombination"], colors=["blue", "orange", "green"],
                  baseline="zero", alpha=0.6)

    plt.xlabel("generation", fontsize=16)
    plt.ylabel("top origins", fontsize=16)
    plt.title("Top 25% new individuals origin comparison", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/top_individuals_origin_comparison_stackplot")
    else:
        plt.show()


def fr8(population, tmp_log):
    relative_diversity_development = tmp_log["relative_diversity_development"]
    plt.figure(figsize=(9, 6))
    plt.plot(relative_diversity_development, linewidth=1.0)
    plt.title("relative diversity development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("relative diversity", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/relative_diversity_development")
    else:
        plt.show()


def fr9(population, tmp_log):
    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_final_population", "wb") as dill_file:
            dill.dump(population, dill_file)


def fr10(population, tmp_log):
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 5.0
    fig = plt.figure(figsize=(18, 13))
    num_param = len(population.individuals[0].genome)
    plt.plot((np.array(tmp_log["max_0_count_of_highest_accuracy_development"])/num_param)*100, label="Top Accuracy Individual", linewidth=5)
    plt.plot((np.array(tmp_log["max_0_count_development"])/num_param)*100, label="Highest Sparsity in Population", linewidth=5)
    plt.plot((np.array(tmp_log["overall_max_0_count_development"])/num_param)*100, label="Highest Sparsity in Evolution", linewidth=5)
    # plt.title("Sparsity Development", fontsize=20)
    plt.xlabel("Generation", fontsize=50)
    plt.ylabel("Sparsity (%)", fontsize=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.legend(fontsize=40)
    plt.grid(visible=True, which='both', linestyle='-', axis='y')
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/sparsity_development")
    else:
        plt.show()


def fr11(population, tmp_log):
    final_population_flattened = {}
    for rank, individual in enumerate(population.individuals):
        final_population_flattened[rank] = {"genome": individual.genome, "origin": individual.origin, "fitness": individual.fitness}

    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_final_population_flattened", "wb") as dill_file:
            dill.dump(final_population_flattened, dill_file)

def fr12(population, tmp_log):
    plt.figure(figsize=(9, 6))
    plt.plot(tmp_log["bound_development"])
    plt.title("accuracy bound development", fontsize=20)
    plt.xlabel("generation", fontsize=16)
    plt.ylabel("accuracy bound", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    # plt.xscale("log")
    if tmp_log.get("log_path"):
        plt.savefig(tmp_log["log_path"] + "/accuracy_bound_development")
    else:
        plt.show()

def fr13(population, tmp_log):
    if tmp_log.get("log_path"):
        if not os.path.isdir(tmp_log["log_path"] + "/pickled_objects"):
            os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_tmp_log", "wb") as dill_file:
            dill.dump(tmp_log, dill_file)



final_routines_dict = {1: (fr1, [ir1, ir11], [er13]), 2: (fr2, [ir5], [er7]), 3: (fr3, [ir6], [er8]),
                       4: (fr4, [], []), 5: (fr5, [], []), 6: (fr6, [ir7], [er9]), 7: (fr7, [ir8], [er10]),
                       8: (fr8, [ir9], [er6]), 9: (fr9, [], []), 10: (fr10, [ir12], [er14]), 11: (fr11, [], []),
                       12: (fr12, [ir14], [er15]), 13: (fr13, [], [])}