from collections import Counter
from typing import List
from sklearn.metrics import accuracy_score
from src.neural_network.subnetwork import Subnetwork
from src.genetic_algorithm.helper_functions import randb

created_genomes = {}


def evaluate_population_method1(population, evaluate_args):
    # Unpack necessary arguments for evaluation
    NN_architecture, model, X_for_accuracy, Y_for_accuracy = get_environment_args(evaluate_args)

    # The individuals with the same fitness value are sorted regarding the number of 0s
    population.individuals.sort(reverse=True,
                                key=lambda individual: (
                                    individual.evaluate(evaluate_accuracy_goal(individual.genome, NN_architecture,
                                                                               model, X_for_accuracy,
                                                                               Y_for_accuracy)),
                                    individual.genome.count(0)))
    return population


def evaluate_population_method2(population, evaluate_args):
    # Unpack necessary arguments for evaluation
    NN_architecture, model, X_for_accuracy, Y_for_accuracy = get_environment_args(evaluate_args)
    population.individuals.sort(reverse=True,
                                key=lambda individual: individual.evaluate(
                                    evaluate_accuracy_goal(individual.genome, NN_architecture, model,
                                                           X_for_accuracy, Y_for_accuracy)))
    return population


def evaluate_population_method3(population, evaluate_args):
    # Unpack necessary arguments for evaluation
    NN_architecture, model, X_for_accuracy, Y_for_accuracy = get_environment_args(evaluate_args)

    # The individuals with the same fitness value are sorted regarding the number of 0s
    population.individuals.sort(reverse=True,
                                key=lambda individual: (
                                    individual.evaluate(accuracy_objective(individual.genome, NN_architecture,
                                                                           model, X_for_accuracy, Y_for_accuracy,
                                                                           use_stored=False)),
                                    individual.genome.count(0)))
    return population


def evaluate_individual_method1(genome, population, evaluate_args):
    NN_architecture, model, X_for_accuracy, Y_for_accuracy = get_environment_args(evaluate_args)
    use_stored = evaluate_args.get("environment_args").get("use_stored", True)
    return evaluate_accuracy_goal(genome, NN_architecture, model, X_for_accuracy, Y_for_accuracy, use_stored)


# ============ methods for bit_vector evaluation ============#

def evaluate_accuracy_goal(bit_vector: List[int], NN_architecture, model, X_for_accuracy, Y_for_accuracy,
                           use_stored=True) -> float:
    """ Needs double_sort to be set for evaluate in population """
    return accuracy_objective(bit_vector, NN_architecture, model, X_for_accuracy, Y_for_accuracy, use_stored)


def evaluate_merged_objective_functions(bit_vector: List[int], NN_architecture, model, population, w1, Y_for_accuracy):
    # Build model from bit_vector
    subnet = Subnetwork(population.individuals[0].genome, NN_architecture, model)

    # calculate first objective function
    f1 = accuracy_score(Y_for_accuracy, subnet.predict())

    # get the maximum 0 count of the individuals of the current population
    max_0_count = 0
    for individual in population.individuals:
        if individual.genome.count(0) > max_0_count:
            max_0_count = individual.genome.count(0)

    # calculate second objective function
    if max_0_count == 0:
        f2 = 1
    else:
        f2 = bit_vector.count(0) / max_0_count

    # combine results of the two objective functions to calculate the overall fitness
    return w1 * f1 + 1 - w1 * f2


def evaluate_goals_alternating_after_n(bit_vector: List[int], NN_architecture, model, population,
                                       evaluate_0_count_generation, X_for_accuracy, Y_for_accuracy):
    # Only use 0 count as fitness measure in multiples of evaluate_0_count_generation
    if population.age % evaluate_0_count_generation == 0:
        return max_0_count_objective(bit_vector, population)
    else:
        return accuracy_objective(bit_vector, NN_architecture, model, X_for_accuracy, Y_for_accuracy)


def evaluate_goals_break_same_accuracy(bit_vector: List[int], NN_architecture, model, population, w1,
                                       X_for_accuracy, Y_for_accuracy):
    # Use the max_0_count goal to invoke change if the half of the population has the same accuracy
    same_fitness = True
    fitness = []
    for individual in population.individuals:
        fitness.append(individual.fitness)
    fitness = dict(Counter(fitness))
    # print(fitness.values(), len(population.individuals))

    if any([value >= len(population.individuals) / 2 for value in fitness.values()]):
        return evaluate_merged_objective_functions(bit_vector, NN_architecture, model, population, w1, Y_for_accuracy)
    else:
        return accuracy_objective(bit_vector, NN_architecture, model, X_for_accuracy, Y_for_accuracy)


def evaluate_choose_random_goal(bit_vector: List[int], NN_architecture, model, X_for_accuracy,
                                Y_for_accuracy, chance, population):
    if randb(chance):
        return accuracy_objective(bit_vector, NN_architecture, model, X_for_accuracy,
                                  Y_for_accuracy)
    else:
        return max_0_count_objective(bit_vector, population)


def evaluate_bit_vector_accuracy(genome, evaluate_args):
    NN_architecture, model, X_for_accuracy, Y_for_accuracy = get_environment_args(evaluate_args)
    return evaluate_accuracy_goal(genome, NN_architecture, model, X_for_accuracy, Y_for_accuracy)


# ============ helper functions ============#

def get_environment_args(evaluate_args):
    NN_architecture = evaluate_args.get("environment_args").get("NN_architecture")
    model = evaluate_args.get("environment_args").get("model")
    X_for_accuracy = evaluate_args.get("environment_args").get("X_for_accuracy")
    Y_for_accuracy = evaluate_args.get("environment_args").get("Y_for_accuracy")
    return NN_architecture, model, X_for_accuracy, Y_for_accuracy


def accuracy_objective(bit_vector: List[int], NN_architecture, model, X_for_accuracy, Y_for_accuracy, use_stored=True):
    # Check if same bit_vector already was evaluated
    if use_stored and created_genomes.get(str(bit_vector)):
        return created_genomes.get(str(bit_vector))
    else:
        # Build model from bit_vector
        subnet = Subnetwork(bit_vector, NN_architecture, model)
        fitness = subnet.get_accuracy(X_for_accuracy, Y_for_accuracy)
        created_genomes[str(bit_vector)] = fitness
        return fitness


def max_0_count_objective(bit_vector: List[int], population):
    # get the maximum 0 count of the individuals of the current population
    max_0_count = 0
    for individual in population.individuals:
        if individual.genome.count(0) > max_0_count:
            max_0_count = individual.genome.count(0)

    if max_0_count == 0:
        return 1
    else:
        return bit_vector.count(0) / max_0_count
