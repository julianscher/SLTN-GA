import math
from copy import copy

from src.neural_network.subnetwork import Subnetwork
from src.experiments.base.init_routines import ir3, ir4, ir2, ir9, ir5, ir7, ir8, ir11, ir6, ir12, ir14
from src.genetic_algorithm.bit_vector_operations import check_if_all_connections_are_part_of_input_output_paths, \
    check_for_floating_connections
from src.genetic_algorithm.helper_functions import calculate_distance_between_individuals_genomes, \
    calculate_relative_diversity, get_mean_population_fitness


def er1(population, tmp_log):
    print("\npopulation.age:", population.age)


def er2(population, tmp_log):
    tmp_log["current_max_0_count"] = 0
    contains_individual_with_greater_max_0_count = False
    for individual in population.individuals:
        if individual.genome.count(0) > tmp_log.get("current_max_0_count"):
            tmp_log["current_max_0_count"] = individual.genome.count(0)
            if tmp_log["current_max_0_count"] > tmp_log.get("max_0_count"):
                tmp_log["max_0_count"] = tmp_log["current_max_0_count"]
                contains_individual_with_greater_max_0_count = True
                tmp_log["max_0_count_last_change"] = 0

    if not contains_individual_with_greater_max_0_count:
        tmp_log["max_0_count_last_change"] += 1


def er3(population, tmp_log):
    tmp_log["distances_to_winning_ticket"] = []
    for individual in population.individuals:
        tmp_log["distances_to_winning_ticket"].append(
            min(calculate_distance_between_individuals_genomes(individual.genome,
                                                               tmp_log["winning_tickets"][0]),
                calculate_distance_between_individuals_genomes(individual.genome,
                                                               tmp_log["winning_tickets"][1])))


def er4(population, tmp_log):
    print("current_max_0_count:", tmp_log["current_max_0_count"])
    print("max_0_count_of_highest_accuracy:", population.individuals[0].genome.count(0))
    print("max_0_count_last_change:", tmp_log["max_0_count_last_change"])
    print("max_0_count:", tmp_log["max_0_count"])
    print("lowest_distance_to_winning_ticket:", min(tmp_log["distances_to_winning_ticket"]))


def er5(population, tmp_log):
    print("current_max_0_count:", tmp_log["current_max_0_count"])
    print("highest_fitness:", population.individuals[0].fitness)
    print("max_0_count_of_highest_accuracy:", population.individuals[0].genome.count(0))
    print("max_0_count_last_change:", tmp_log["max_0_count_last_change"])
    print("max_0_count:", tmp_log["max_0_count"])


def er6(population, tmp_log):
    relative_diversity = calculate_relative_diversity(population, tmp_log["max_diversity"])
    # print("population_relative_diversity:", "{0:,.3f}".format(relative_diversity))
    tmp_log["relative_diversity_development"].append(relative_diversity)


def er7(population, tmp_log):
    NN_architecture = population.problem_instance.NN_architecture
    model = population.problem_instance.model
    subnet = Subnetwork(population.individuals[0].genome, NN_architecture, model)
    tmp_log["loss_development"].append(subnet.get_loss(population.problem_instance.args["X_for_accuracy"],
                                                       population.problem_instance.args["Y_for_accuracy"]))


def er8(population, tmp_log):
    NN_architecture = population.problem_instance.NN_architecture
    model = population.problem_instance.model
    subnet = Subnetwork(population.individuals[0].genome, NN_architecture, model)
    tmp_log["accuracy_development"].append(
        subnet.get_accuracy(population.problem_instance.args["X_for_accuracy"],
                            population.problem_instance.args["Y_for_accuracy"]))


def er9(population, tmp_log):
    # Get all selected individuals that were created via recombination in this generation
    children = []
    for individual in population.individuals:
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
    tmp_log["goodness_of_children"].append((better_both_count, better_one_count, worse_count))


def er10(population, tmp_log):
    """ Determine through which variational operation, the top 25% of new individuals were created """
    # Get all the individuals that were created in this generation
    new_individuals = []
    for individual in population.individuals:
        if individual.age == 0:
            new_individuals.append(individual)

    # Get the top 25% of these individuals (they are already ordered regarding their fitness through select())
    top_individuals = new_individuals[:math.ceil(len(new_individuals) * 0.25)]
    tmp_log["top_recombine_counts"].append(
        sum([1 for individual in top_individuals if individual.origin == "recombine"]))
    tmp_log["top_mutate_counts"].append(sum([1 for individual in top_individuals if individual.origin == "mutate"]))
    tmp_log["top_migrate_counts"].append(sum([1 for individual in top_individuals if individual.origin == "migrate"]))


def er11(population, tmp_log):
    """ Determine which operation had the highest impact on the improvement of the overall fitness (here: accuracy)
    of the population in this generation """
    population_fitness = get_mean_population_fitness(population)
    pass


def er12(population, tmp_log):
    # Make sure, that at every generation only individuals with the correct connectivity strictness level exist
    NN_architecture = population.problem_instance.NN_architecture
    connectivity_scheme = population.args.get("recombine_args").get("connectivity_scheme")
    if connectivity_scheme == "strict":
        for individual in population.individuals:
            if not check_if_all_connections_are_part_of_input_output_paths(NN_architecture, individual.genome):
                print(individual)
                raise ValueError("Connectivity strictness level \"strict\" hurt")
    elif connectivity_scheme == "soft":
        for individual in population.individuals:
            if check_for_floating_connections(NN_architecture, individual.genome):
                print(individual)
                raise ValueError("Connectivity strictness level \"soft\" hurt")


def er13(population, tmp_log):
    for individual in population.individuals:
        if tmp_log["genomes_occurrences"].get(str(individual.genome)):
            tmp_log["genomes_occurrences"][str(individual.genome)] += 1
        else:
            tmp_log["genomes_occurrences"][str(individual.genome)] = 1


def er14(population, tmp_log):
    tmp_log["max_0_count_of_highest_accuracy_development"].append(population.individuals[0].genome.count(0))
    max_0_count = 0
    for individual in population.individuals:
        if individual.genome.count(0) > max_0_count:
            max_0_count = individual.genome.count(0)
    tmp_log["max_0_count_development"].append(max_0_count)
    if len(tmp_log["overall_max_0_count_development"]) == 0:
        tmp_log["overall_max_0_count_development"].append(max_0_count)
    elif max_0_count > tmp_log["overall_max_0_count_development"][-1]:
        tmp_log["overall_max_0_count_development"].append(max_0_count)
    else:
        tmp_log["overall_max_0_count_development"].append(tmp_log["overall_max_0_count_development"][-1])


def er15(population, tmp_log):
    print("accuracy_bound:", population.args.get("generate_args").get("bound"))
    tmp_log["bound_development"].append(copy(population.args.get("generate_args").get("bound")))

# If er has dependency to other er, make sure, that dependent er is executed after the dependency
# -> dependent number must be higher than number of dependency
# List all dependencies, even for er-dependencies
evol_routines_dict = {1: (er1, [], []), 2: (er2, [ir3, ir4], []), 3: (er3, [ir2], []),
                      4: (er4, [ir2, ir3, ir4], [er2, er3]), 5: (er5, [ir3, ir4], [er2]), 6: (er6, [ir9], []),
                      7: (er7, [ir5], []), 8: (er8, [ir6], []), 9: (er9, [ir7], []), 10: (er10, [ir8], []),
                      11: (er11, [], []), 12: (er12, [], []), 13: (er13, [ir11], []), 14: (er14, [ir12], []),
                      15: (er15, [ir14], [])}