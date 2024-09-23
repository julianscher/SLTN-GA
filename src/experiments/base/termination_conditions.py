from src.experiments.base.evol_routines import er2, er8
from src.experiments.base.init_routines import ir4, ir3, ir6
from src.genetic_algorithm.helper_functions import winning_ticket_included, calculate_diversity


def t1(population, tmp_log):
    if winning_ticket_included(population) or population.age == 100:
        return True


def t2(population, tmp_log):
    if population.individuals[0].fitness and population.individuals[0].fitness >= 0.9 \
            and tmp_log["max_0_count_last_change"] >= 100 or tmp_log["max_0_count_last_change"] >= 100 \
            and calculate_diversity(population) == 0:
        return True


def t3(population, tmp_log):
    if population.age == 50:
        return True


def t4(population, tmp_log):
    # The evolution should take less than 200 generations
    if len(tmp_log["accuracy_development"]) >= 200:
        return True
    else:
        if len(tmp_log["accuracy_development"]) >= 100:
            # If there was no change in accuracy in the last 50 generations, terminate
            if tmp_log["accuracy_development"][-50:].count(tmp_log["accuracy_development"][-1]) == 50:
                return True


def t5(population, tmp_log):
    # The evolution should take less than 200 generations
    if len(tmp_log["accuracy_development"]) >= 200:
        return True
    else:
        if len(tmp_log["accuracy_development"]) >= 100:
            # terminate if last accuracy is the best found so far
            if tmp_log["accuracy_development"][-1] == max(tmp_log["accuracy_development"]):
                return True


def t6(population, tmp_log):
    if population.individuals[0].fitness and population.individuals[0].fitness == 1.0:
        return True


def t7(population, tmp_log):
    """ For embedding experiment situation1 """
    if population.individuals[0].fitness and population.individuals[0].fitness >= 0.9998125 and \
        population.individuals[0].genome.count(0) >= 24:
        return True


def t8(population, tmp_log):
    """ For embedded experiment situation2 """
    if population.age >= 1000 or population.individuals[0].fitness and population.individuals[0].fitness >= 0.99975 and \
        population.individuals[0].genome.count(0) >= 588:
        return True


def t9(population, tmp_log):
    """ For embedding experiment situation3 """
    if population.age >= 1000 or population.individuals[0].fitness and population.individuals[0].fitness >= 0.9998125 and \
            population.individuals[0].genome.count(0) >= 75:
        return True


def t10(population, tmp_log):
    if len(tmp_log["accuracy_development"]) >= 100:
        # If there was no change in accuracy in the last 50 generations, terminate
        if tmp_log["accuracy_development"][-50:].count(tmp_log["accuracy_development"][-1]) == 50:
            return True


termination_conditions_dict = {1: (t1, [], []), 2: (t2, [ir3, ir4], [er2]), 3: (t3, [], []), 4: (t4, [ir6], [er8]),
                               5: (t5, [ir6], [er8]), 6: (t6, [], []), 7: (t7, [], []), 8: (t8, [], []),
                               9: (t9, [], []), 10: (t10, [ir6], [er8])}
