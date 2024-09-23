import os
import time
from copy import copy
import dill
from src.genetic_algorithm.helper_functions import calculate_max_diversity, calculate_relative_diversity


def ir1(population, tmp_log):
    tmp_log["start_time"] = time.time()


def ir2(population, tmp_log):
    tmp_log["winning_tickets"] = [
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
         1, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                                   0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]]


def ir3(population, tmp_log):
    tmp_log["max_0_count"] = 0


def ir4(population, tmp_log):
    tmp_log["max_0_count_last_change"] = 0


def ir5(population, tmp_log):
    tmp_log["loss_development"] = []


def ir6(population, tmp_log):
    tmp_log["accuracy_development"] = []


def ir7(population, tmp_log):
    tmp_log["goodness_of_children"] = []


def ir8(population, tmp_log):
    tmp_log["top_recombine_counts"] = []
    tmp_log["top_mutate_counts"] = []
    tmp_log["top_migrate_counts"] = []
    tmp_log["top_generate_counts"] = []


def ir9(population, tmp_log):
    tmp_log["max_diversity"] = calculate_max_diversity(population.size,
                                                       [0 for _ in range(len(population.individuals[0].genome))],
                                                       [1 for _ in range(len(population.individuals[0].genome))])
    tmp_log["relative_diversity_development"] = [calculate_relative_diversity(population, tmp_log["max_diversity"])]


def ir10(population, tmp_log):
    population.evaluate_by_accuracy()
    print(population.__str__())


def ir11(population, tmp_log):
    tmp_log["genomes_occurrences"] = {}


def ir12(population, tmp_log):
    tmp_log["max_0_count_of_highest_accuracy_development"] = []
    tmp_log["max_0_count_development"] = []
    tmp_log["overall_max_0_count_development"] = []


def ir13(population, tmp_log):
    if tmp_log.get("log_path"):
        os.mkdir(tmp_log["log_path"] + "/pickled_objects")
        with open(tmp_log["log_path"] + "/pickled_objects/pickled_initial_population", "wb") as dill_file:
            dill.dump(population, dill_file)


def ir14(population, tmp_log):
    tmp_log["bound_development"] = [copy(population.args.get("generate_args").get("bound", 0.85))]


init_routines_dict = {1: ir1, 2: ir2, 3: ir3, 4: ir4, 5: ir5, 6: ir6, 7: ir7, 8: ir8,
                      9: ir9, 10: ir10, 11: ir11, 12: ir12, 13: ir13, 14: ir14}