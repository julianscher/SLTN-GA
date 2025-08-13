import copy

from genetic_algorithm.logging.evol_routines import er2, er8
from genetic_algorithm.logging.init_routines import ir3, ir4, ir6
from models.fc.mlp_embedded import MLPEmbedded
from utilities.ga_utils import winning_ticket_included, calculate_diversity, calculate_relative_diversity
from utilities.net_utils import accuracy


def t1(ga, tmp_log, termination_condition_args):
    if winning_ticket_included(ga.population) or ga.population.age == 100:
        return True


def t2(ga, tmp_log, termination_condition_args):
    if ga.population.individuals[0].fitness and ga.population.individuals[0].fitness >= 0.9 \
            and tmp_log["max_0_count_last_change"] >= 100 or tmp_log["max_0_count_last_change"] >= 100 \
            and calculate_diversity(ga.population) == 0:
        return True


def t3(ga, tmp_log, termination_condition_args):
    population_age = termination_condition_args.get("population_age", 50)
    if ga.population.age == population_age:
        return True


def t4(ga, tmp_log, termination_condition_args):
    # The evolution should take less than 200 generations
    if len(tmp_log["accuracy_development"]) >= 200:
        return True
    else:
        if len(tmp_log["accuracy_development"]) >= 100:
            # If there was no change in accuracy in the last 50 generations, terminate
            if tmp_log["accuracy_development"][-50:].count(tmp_log["accuracy_development"][-1]) == 50:
                return True


def t5(ga, tmp_log, termination_condition_args):
    # The evolution should take less than 200 generations
    if len(tmp_log["accuracy_development"]) >= 200:
        return True
    else:
        if len(tmp_log["accuracy_development"]) >= 100:
            # terminate if last accuracy is the best found so far
            if tmp_log["accuracy_development"][-1] == max(tmp_log["accuracy_development"]):
                return True


def t6(ga, tmp_log, termination_condition_args):
    if ga.population.individuals[0].fitness and ga.population.individuals[0].fitness == 1.0:
        return True


def t7(ga, tmp_log, termination_condition_args):
    population_age = termination_condition_args.get("population_age")
    no_change_in = termination_condition_args.get("no_change_in")
    if ga.population.age >= 100:
        if population_age and ga.population.age >= population_age:
            return True
        # If there was no change in accuracy in the last x logs, terminate
        accuracies = list(tmp_log["accuracy_development"].values())
        if len(accuracies) >= no_change_in and accuracies[-no_change_in:].count(accuracies[-1]) == no_change_in:
            return True


def t8(ga, tmp_log, termination_condition_args):
    """
    Terminates GA evolution if (pruned) subnetwork has been found.
    This method should work if all parameters except those of the trained network have been initialized with 0s.
    """
    model = ga.model
    assert isinstance(model, MLPEmbedded)

    # Check if the only remaining active connections are part of the embedded trained network.
    # This takes into account that the GA might have pruned some of the connections of the trained network.
    for idx, bit in enumerate(model.embedded_subnetwork_bitvector):
        if bit == 0 and ga.population.individuals[0].genome[idx] != 0:
            return False

    return True


def t9(ga, tmp_log, termination_condition_args):
    """ Terminate if population diversity falls under 0.1 """
    if calculate_relative_diversity(ga.population, tmp_log["max_diversity"]) <= 0.1:
        return True

def t10(ga, tmp_log, termination_condition_args):
    population_age = termination_condition_args.get("population_age")
    no_change_in = termination_condition_args.get("no_change_in")
    individual_evaluation = copy.deepcopy(ga.population.ga_config.individual_evaluation)
    individual_evaluation.quality_metric = accuracy
    individual_evaluation.performance_maximizing = True
    fitness = individual_evaluation.evaluate_individual(ga.population.individuals[0].genome, None)
    if fitness == 100:
        print("1")
        return True
    elif ga.population.age >= 100:
        if population_age and ga.population.age >= population_age:
            print("2")
            return True
        # If there was no change in accuracy in the last x logs, terminate
        accuracies = list(tmp_log["accuracy_development"].values())
        if len(accuracies) >= no_change_in and accuracies[-no_change_in:].count(accuracies[-1]) == no_change_in:
            print("3")
            return True



termination_conditions_dict = {1: (t1, [], []), 2: (t2, [ir3, ir4], [er2]), 3: (t3, [], []), 4: (t4, [ir6], [er8]),
                               5: (t5, [ir6], [er8]), 6: (t6, [], []), 7: (t7, [ir6], [er8]), 8: (t8, [], []),
                               9: (t9, [], []), 10: (t10, [], [])}
