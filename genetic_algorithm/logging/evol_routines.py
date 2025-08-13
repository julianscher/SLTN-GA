import math
from copy import copy

import numpy as np

from genetic_algorithm.logging.init_routines import ir3, ir4, ir2, ir5, ir6, ir9, ir8, ir7, ir11, ir12, ir19, ir18, \
    ir17, ir14
from utilities.fc.fc_bit_vector_operations import check_if_all_connections_are_part_of_input_output_paths, \
    check_for_floating_connections
from utilities.fc.fc_utils import get_NN_architecture_from_model
from utilities.ga_utils import calculate_distance_between_individuals_genomes, \
    calculate_relative_diversity, get_mean_population_fitness
from utilities.net_utils import accuracy


def er1(ga, tmp_log):
    print("\npopulation.age:", ga.population.age)


def er2(ga, tmp_log):
    tmp_log["current_max_0_count"] = 0
    contains_individual_with_greater_max_0_count = False
    for individual in ga.population.individuals:
        if np.count_nonzero(individual.genome == 0) > tmp_log.get("current_max_0_count"):
            tmp_log["current_max_0_count"] = np.count_nonzero(individual.genome == 0)
            if tmp_log["current_max_0_count"] > tmp_log.get("max_0_count"):
                tmp_log["max_0_count"] = tmp_log["current_max_0_count"]
                contains_individual_with_greater_max_0_count = True
                tmp_log["max_0_count_last_change"] = 0

    if not contains_individual_with_greater_max_0_count:
        tmp_log["max_0_count_last_change"] += 1


def er3(ga, tmp_log):
    generation = ga.population.age
    tmp_log["distances_to_winning_ticket"][generation] = []
    for individual in ga.population.individuals:
        tmp_log["distances_to_winning_ticket"][generation] = \
            min(calculate_distance_between_individuals_genomes(individual.genome,
                                                               tmp_log["winning_tickets"][0]),
                calculate_distance_between_individuals_genomes(individual.genome,
                                                               tmp_log["winning_tickets"][1]))


def er4(ga, tmp_log):
    print("current_max_0_count:", tmp_log["current_max_0_count"])
    print("max_0_count_of_highest_accuracy:", ga.population.individuals[0].genome.count(0))
    print("max_0_count_last_change:", tmp_log["max_0_count_last_change"])
    print("max_0_count:", tmp_log["max_0_count"])
    print("lowest_distance_to_winning_ticket:", min(tmp_log["distances_to_winning_ticket"]))


def er5(ga, tmp_log):
    print("current_max_0_count:", tmp_log["current_max_0_count"])
    print("highest_fitness:", ga.population.individuals[0].fitness)
    print("max_0_count_of_highest_accuracy:", np.count_nonzero(ga.population.individuals[0].genome == 0))
    print("max_0_count_last_change:", tmp_log["max_0_count_last_change"])
    print("max_0_count:", tmp_log["max_0_count"])


def er6(ga, tmp_log):
    generation = ga.population.age
    relative_diversity = calculate_relative_diversity(ga.population, tmp_log["max_diversity"])
    # print("population_relative_diversity:", "{0:,.3f}".format(relative_diversity))
    tmp_log["relative_diversity_development"][generation] = relative_diversity


def er7(ga, tmp_log):
    generation = ga.population.age
    criterion = ga.criterion
    evaluation = ga.population.ga_config.individual_evaluation
    initial_quality_metric = evaluation.quality_metric
    evaluation.quality_metric = criterion
    loss = evaluation.evaluate_individual(ga.population.individuals[0].genome, None)
    evaluation.quality_metric = initial_quality_metric

    print(loss)
    tmp_log["loss_development"][generation] = loss


def er8(ga, tmp_log):
    generation = ga.population.age
    quality = ga.population.ga_config.evaluate_individual(ga.population.individuals[0].genome)
    tmp_log["accuracy_development"][generation] = quality


def er9(ga, tmp_log):
    generation = ga.population.age
    # Get all selected individuals that were created via recombination in this generation
    children = []
    for individual in ga.population.individuals:
        if individual.origin == "recombine" and individual.age == 0:
            children.append(individual)

    # Check how often the child was better than one or both of its parents
    better_both_count = 0
    better_one_count = 0
    worse_count = 0
    for child in children:
        if child.fitness > child.parents[0].fitness and child.fitness > child.parents[1].fitness:
            better_both_count += 1
        elif child.fitness > child.parents[0].fitness or child.fitness > child.parents[1].fitness:
            better_one_count += 1
        else:
            worse_count += 1
    tmp_log["goodness_of_children"][generation] = (better_both_count, better_one_count, worse_count)


def er10(ga, tmp_log):
    """ Determine through which variational operation, the top 25% of new individuals were created """
    generation = ga.population.age
    # Get all the individuals that were created in this generation
    new_individuals = []
    for individual in ga.population.individuals:
        if individual.age == 0:
            new_individuals.append(individual)

    # Get the top 25% of these individuals (they are already ordered regarding their fitness through select())
    top_individuals = new_individuals[:math.ceil(len(new_individuals) * 0.25)]
    tmp_log["top_recombine_counts"][generation] = \
        sum([1 for individual in top_individuals if individual.origin == "recombine"])
    tmp_log["top_mutate_counts"][generation] = sum([1 for individual in top_individuals if individual.origin == "mutate"])
    tmp_log["top_migrate_counts"][generation] = sum([1 for individual in top_individuals if individual.origin == "migrate"])


def er11(ga, tmp_log):
    """ Determine which operation had the highest impact on the improvement of the overall fitness (here: accuracy)
    of the population in this generation """
    population_fitness = get_mean_population_fitness(ga.population)

def er12(ga, tmp_log):
    # Make sure, that at every generation only individuals with the correct connectivity strictness level exist
    model = ga.model
    nn_architecture = get_NN_architecture_from_model(model)
    connectivity_scheme = ga.population.ga_config.recombination.connectivity_scheme
    if connectivity_scheme == "strict":
        for individual in ga.population.individuals:
            if not check_if_all_connections_are_part_of_input_output_paths(nn_architecture, individual.genome):
                print(individual)
                raise ValueError("Connectivity strictness level \"strict\" hurt")
    elif connectivity_scheme == "soft":
        for individual in ga.population.individuals:
            if check_for_floating_connections(nn_architecture, individual.genome):
                print(individual)
                raise ValueError("Connectivity strictness level \"soft\" hurt")

def er13(ga, tmp_log):
    generation = ga.population.age
    for individual in ga.population.individuals:
        if tmp_log["genomes_occurrences"].get(str(individual.genome)):
            tmp_log["genomes_occurrences"][str(individual.genome)] += 1
        else:
            tmp_log["genomes_occurrences"][str(individual.genome)] = 1


def er14(ga, tmp_log):
    generation = ga.population.age
    tmp_log["max_0_count_of_highest_accuracy_development"][generation] = np.count_nonzero(ga.population.individuals[0].genome == 0)
    max_0_count = 0
    for individual in ga.population.individuals:
        if np.count_nonzero(individual.genome == 0) > max_0_count:
            max_0_count = np.count_nonzero(individual.genome == 0)
    tmp_log["max_0_count_development"][generation] = max_0_count
    if len(tmp_log["overall_max_0_count_development"].keys()) == 0:
        tmp_log["overall_max_0_count_development"][generation] = max_0_count
    elif max_0_count > tmp_log["overall_max_0_count_development"][list(tmp_log["overall_max_0_count_development"].keys())[-1]]:
        tmp_log["overall_max_0_count_development"][generation] = max_0_count
    else:
        tmp_log["overall_max_0_count_development"][generation] = tmp_log["overall_max_0_count_development"][list(tmp_log["overall_max_0_count_development"].keys())[-1]]


def er15(ga, tmp_log):
    generation = ga.population.age
    bound = ga.population.ga_config.generation.bound
    print("accuracy_bound:", bound)
    tmp_log["bound_development"][generation] = copy(bound)

def er16(ga, tmp_log):
    generation = ga.population.age
    best_individual = ga.population.individuals[0].genome

    subnet = ga.model.apply_mask(best_individual)
    X_train = ga.train_loader.dataset.tensors[0]
    y_train = ga.train_loader.dataset.tensors[1]
    train_accuracy = accuracy(subnet(X_train), y_train, ga.device)
    tmp_log["train_accuracy_development"][generation] = train_accuracy
    print("training-accuracy:", train_accuracy)

def er17(ga, tmp_log):
    generation = ga.population.age
    best_individual = ga.population.individuals[0].genome

    subnet = ga.model.apply_mask(best_individual)
    X_val = ga.val_loader.dataset.tensors[0]
    y_val = ga.val_loader.dataset.tensors[1]
    val_accuracy = accuracy(subnet(X_val), y_val, ga.device)
    tmp_log["val_accuracy_development"][generation] = val_accuracy
    print("validation-accuracy:", val_accuracy)


def er18(ga, tmp_log):
    generation = ga.population.age
    best_individual = ga.population.individuals[0].genome

    subnet = ga.model.apply_mask(best_individual)
    X_test = ga.test_loader.dataset.tensors[0]
    y_test = ga.test_loader.dataset.tensors[1]
    test_accuracy = accuracy(subnet(X_test), y_test, ga.device)
    tmp_log["test_accuracy_development"][generation] = test_accuracy
    print("test-accuracy:", test_accuracy)


# If er has dependency to other er, make sure, that dependent er is executed after the dependency
# -> dependent number must be higher than number of dependency
# List all dependencies, even for er-dependencies
evol_routines_dict = {1: (er1, [], []), 2: (er2, [ir3, ir4], []), 3: (er3, [ir2], []),
                      4: (er4, [ir2, ir3, ir4], [er2, er3]), 5: (er5, [ir3, ir4], [er2]), 6: (er6, [ir9], []),
                      7: (er7, [ir5], []), 8: (er8, [ir6], []), 9: (er9, [ir7], []), 10: (er10, [ir8], []),
                      11: (er11, [], []), 12: (er12, [], []), 13: (er13, [ir11], []), 14: (er14, [ir12], []),
                      15: (er15, [ir14], []), 16: (er16, [ir17], []), 17: (er17, [ir18], []), 18: (er18, [ir19], []),}