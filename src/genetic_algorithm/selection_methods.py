from src.genetic_algorithm.helper_functions import randf

# ============ Selection methods ============#


def select_by_cutoff(pop, size):
    """ select all individuals of population until cut-off """
    return pop[0:size]


def select_by_graceful_cutoff(pop, size, age_grace_period: int = 5):
    """ select all individuals of population until cut-off, but retain
        all individuals younger than the grace-period age """
    new_pop = []
    for individual in pop:
        if individual.age >= 1 and individual.age <= age_grace_period:
            new_pop.append(individual)
    for new_individual in new_pop:
        pop.remove(new_individual)
    new_pop += pop
    return new_pop[0:size]


def better_select_by_roulette(pop, size):
    """ select individuals of population with probability proportional to
        their fitness """
    new_pop = []
    while len(new_pop) < size:
        if len(new_pop) > 0:
            if new_pop[-1] in pop:
                pop.remove(new_pop[-1])
        total_fitness = sum([individual.get_fitness() for individual in pop])
        ball = randf() * total_fitness
        current_fitness  = 0
        for individual in pop:
            current_fitness += individual.get_fitness()
            if ball < current_fitness:
                new_pop.append(individual)
                break
    return new_pop


def better_select_by_walk(pop, size, walk_rate: float = 0.5):
    """ select individuals of population if random draw is smaller than
        some probability-rate (like walking a probabilistic tree structure) """
    new_pop = []
    while len(new_pop) < size:
        if len(new_pop) > 0:
            if new_pop[-1] in pop:
                pop.remove(new_pop[-1])
        for individual in pop:
            if randf() < walk_rate:
                new_pop.append(individual)
                break
    return new_pop
