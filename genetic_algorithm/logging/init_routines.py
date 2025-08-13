import os
import time
from copy import copy, deepcopy

import dill
import torch

from utilities.ga_utils import calculate_max_diversity, calculate_relative_diversity


def ir1(ga, tmp_log):
    tmp_log["start_time"] = time.time()


def ir2(ga, tmp_log):
    tmp_log["winning_tickets"] = [
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
         1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                                   0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]]


def ir3(ga, tmp_log):
    tmp_log["max_0_count"] = 0


def ir4(ga, tmp_log):
    tmp_log["max_0_count_last_change"] = 0


def ir5(ga, tmp_log):
    tmp_log["loss_development"] = {}


def ir6(ga, tmp_log):
    tmp_log["accuracy_development"] = {}


def ir7(ga, tmp_log):
    tmp_log["goodness_of_children"] = {}


def ir8(ga, tmp_log):
    tmp_log["top_recombine_counts"] = {}
    tmp_log["top_mutate_counts"] = {}
    tmp_log["top_migrate_counts"] = {}
    tmp_log["top_generate_counts"] = {}


def ir9(ga, tmp_log):
    tmp_log["max_diversity"] = calculate_max_diversity(ga.population.size,
                                                       [0 for _ in range(len(ga.population.individuals[0].genome))],
                                                       [1 for _ in range(len(ga.population.individuals[0].genome))])
    tmp_log["relative_diversity_development"] = {0:calculate_relative_diversity(ga.population, tmp_log["max_diversity"])}


def ir10(ga, tmp_log):
    ga.population.evaluate_survivors()
    print(ga.population.__str__())


def ir11(ga, tmp_log):
    tmp_log["genomes_occurrences"] = {}


def ir12(ga, tmp_log):
    tmp_log["max_0_count_of_highest_accuracy_development"] = {}
    tmp_log["max_0_count_development"] = {}
    tmp_log["overall_max_0_count_development"] = {}


def ir13(ga, tmp_log):
    if tmp_log.get("log_path"):
        os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_initial_population", "wb") as dill_file:
            dill.dump(ga.population, dill_file)


def ir14(ga, tmp_log):
    tmp_log["bound_development"] = {0:[copy(ga.population.ga_config.generation.bound)]}


def ir15(ga, tmp_log):
    tmp_log["genetic_algorithm"] = deepcopy(ga)


def ir16(ga, tmp_log):
    """ Requires ir13 to be called before"""
    torch.save(ga.model.net.state_dict(), tmp_log["log_path"] + "/model_init.pt")

def ir17(ga, tmp_log):
    tmp_log["train_accuracy_development"] = {}

def ir18(ga, tmp_log):
    tmp_log["val_accuracy_development"] = {}

def ir19(ga, tmp_log):
    tmp_log["test_accuracy_development"] = {}


init_routines_dict = {1: ir1, 2: ir2, 3: ir3, 4: ir4, 5: ir5, 6: ir6, 7: ir7, 8: ir8,
                      9: ir9, 10: ir10, 11: ir11, 12: ir12, 13: ir13, 14: ir14, 15: ir15, 16: ir16,
                      17: ir17, 18: ir18, 19: ir19}